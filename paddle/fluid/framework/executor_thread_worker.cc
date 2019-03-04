/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/executor_thread_worker.h"
#include <algorithm>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/framework/eigen.h"

namespace paddle {
namespace framework {

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void ExecutorThreadWorker::CreateThreadScope(const ProgramDesc& program) {
  auto& block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_, "root_scope should be set before creating thread scope");

  const std::vector<std::string>& input_feed = readers_[0]->GetUseSlotAlias();
  for (auto& s : input_feed) {
    if (s != "click") {
      ids_names_.push_back(s);
    }
  }

  thread_scope_.reset(new Scope);
  param_scope_.reset(new Scope);
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      const std::string var_name = var->Name();
      PADDLE_ENFORCE(var_name != "embedding", "var->Name() == \"embedding\"");

      param_names_.push_back(var_name);

      auto& root_tensor = root_scope_->FindVar(var_name)->Get<LoDTensor>();
      auto* thread_tensor = thread_scope_->Var(var_name)->GetMutable<LoDTensor>();
      thread_tensor->Resize(root_tensor.dims());
      cudaMemcpy(thread_tensor->mutable_data<float>(gpu_place_),
          root_tensor.data<float>(),
          root_tensor.numel() * sizeof(float),
          cudaMemcpyHostToDevice);

      auto* param_tensor = param_scope_->Var(var_name)->GetMutable<LoDTensor>();
      param_tensor->Resize(root_tensor.dims());
      cudaMemcpy(param_tensor->mutable_data<float>(gpu_place_),
          root_tensor.data<float>(),
          root_tensor.numel() * sizeof(float),
          cudaMemcpyHostToDevice);

      auto* delta_tensor = param_scope_->Var(var_name + "@DELTA")->GetMutable<LoDTensor>();
      delta_tensor->Resize(root_tensor.dims());
    }
  }

  scope_pool_.reset(new operators::reader::BlockingQueue<Scope*>(nscopes_));
  emb_ff_scope_queue_.reset(new operators::reader::BlockingQueue<Scope*>(nscopes_));
  emb_bp_scope_queue_.reset(new operators::reader::BlockingQueue<Scope*>(nscopes_));
  for (int i = 0; i < nscopes_; ++i) {
    Scope* scope = &thread_scope_->NewScope();
    for (auto& var : block.AllVars()) {
      if (!var->Persistable()) {
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }
    scope_pool_->Send(scope);
  }

  reader_queue_.reset(new operators::reader::BlockingQueue<DataFeed*>(readers_.size()));
  for (auto& reader : readers_) {
    reader_queue_->Send(reader.get());
  }
}

void ExecutorThreadWorker::CreateThreadOperators(const ProgramDesc& program) {
  auto& block = program.Block(0);
  op_names_.clear();
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase* local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
  }
}

void ExecutorThreadWorker::CreateThreadResource(const framework::ProgramDesc& program) {
  CreateThreadScope(program);
  CreateThreadOperators(program);
}

template <typename T>
void print_lod_tensor(std::string var_name, const LoDTensor& lod_tensor) {
  auto inspect = lod_tensor.data<T>();
  auto element_num = lod_tensor.numel();

  std::ostringstream sstream;
  sstream << var_name << " (element num " << element_num << "): [";
  sstream << inspect[0];
  for (int j = 1; j < element_num; ++j) {
    sstream << " " << inspect[j];
  }
  sstream << "]";

  std::cout << sstream.str() << std::endl;
}

static void print_fetch_var(Scope* scope, const std::string& var_name) {
  auto& tensor = scope->FindVar(var_name)->Get<LoDTensor>();

#define PrintLoDTensorCallback(cpp_type, proto_type) \
  do {                                               \
    if (tensor.type() == proto_type) {               \
      print_lod_tensor<cpp_type>(var_name, tensor);  \
      return;                                        \
    }                                                \
  } while (0)

  _ForEachDataType_(PrintLoDTensorCallback);
  VLOG(1) << "print_fetch_var: unrecognized data type:" << tensor.type();
}

static const int user_slot_num = 73;
static const int news_slot_num = 44;
void ExecutorThreadWorker::LookupTableSumConcat(Scope* scope) {
  operators::math::BlasT<platform::CPUDeviceContext, float> blas(*cpu_dev_ctx_);
  const LoDTensor& embedding = root_scope_->FindVar("embedding")->Get<LoDTensor>();
  const float* table_data = embedding.data<float>();
  const DDim& dims = embedding.dims();
  PADDLE_ENFORCE(dims.size() == 2, "embedding dims size is not equel to 2");
  const int voc_size = dims[0];
  const int emb_size = dims[1];

  const LoD& lod = scope->FindVar(ids_names_[0])->Get<LoDTensor>().lod();
  const int batch_size = lod[0].size() - 1;
  PADDLE_ENFORCE(ids_names_.size() == user_slot_num + news_slot_num, "mismatch");
  LoDTensor* user_concat = scope->Var("user_concat_cpu")->GetMutable<LoDTensor>();
  LoDTensor* news_concat = scope->Var("news_concat_cpu")->GetMutable<LoDTensor>();
  user_concat->Resize({batch_size, user_slot_num * emb_size});
  news_concat->Resize({batch_size, news_slot_num * emb_size});
  float* user_concat_data = user_concat->mutable_data<float>(cpu_place_);
  float* news_concat_data = news_concat->mutable_data<float>(cpu_place_);
  memset(user_concat_data, 0, user_concat->memory_size());
  memset(news_concat_data, 0, news_concat->memory_size());

  float* concat_data = user_concat_data;
  for (size_t i = 0; i < ids_names_.size(); ++i) {
    if (i >= user_slot_num) {
       concat_data = news_concat_data;
    }
    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
    const int64_t* ids_data = ids.data<int64_t>();
    const int numel = ids.numel();
    for (int j = 0; j < numel; ++j) {
      if (ids_data[j] == padding_idx_) {
        continue;
      }
      PADDLE_ENFORCE(ids_data[j] < voc_size, "Input id is out of the range of the vocabulary");
      blas.VADD(emb_size, table_data + emb_size * ids_data[j], concat_data + emb_size * (i % user_slot_num), concat_data + emb_size * (i % user_slot_num));
    }
  }
}

void ExecutorThreadWorker::LookupTableSumConcatGrad(Scope* scope) {
  operators::math::BlasT<platform::CPUDeviceContext, float> blas(*cpu_dev_ctx_);
  LoDTensor* embedding = root_scope_->FindVar("embedding")->GetMutable<LoDTensor>();
  float* table_data = embedding->mutable_data<float>(cpu_place_);
  const size_t emb_size = embedding->dims()[1];

  const LoDTensor& user_concat = scope->FindVar("user_concat_grad_cpu")->Get<LoDTensor>();
  const LoDTensor& news_concat = scope->FindVar("news_concat_grad_cpu")->Get<LoDTensor>();
  const float* user_concat_cpu_data = user_concat.data<float>();
  const float* news_concat_cpu_data = news_concat.data<float>();

  const float* concat_cpu_data = user_concat_cpu_data;
  for (size_t i = 0; i < ids_names_.size(); ++i) {
    if (i >= user_slot_num) {
      concat_cpu_data = news_concat_cpu_data;
    }

    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
    const int64_t* ids_data = ids.data<int64_t>();
    const int numel = ids.numel();
    for (int j = 0; j < numel; ++j) {
      if (ids_data[j] == padding_idx_) {
        continue;
      }
      // update embedding
      blas.VADD(emb_size, concat_cpu_data + (i % user_slot_num), table_data + emb_size * ids_data[j], table_data + emb_size * ids_data[j]);
    }
  }
}

//void ExecutorThreadWorker::LookupTable(Scope* scope) {
//  const LoDTensor& embedding = root_scope_->FindVar("embedding")->Get<LoDTensor>();
//  const float* table_data = embedding.data<float>();
//  const DDim& dims = embedding.dims();
//  PADDLE_ENFORCE(dims.size() == 2, "embedding dims size is not equel to 2");
//  const int voc_size = dims[0];
//  const int emb_size = dims[1];
//  for (size_t i = 0; i < ids_names_.size(); ++i) {
//    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
//    LoDTensor* emb_cpu = scope->Var(emb_cpu_names_[i])->GetMutable<LoDTensor>();
//    emb_cpu->set_lod(ids.lod());
//    DDim d = ids.dims();
//    d[1] = emb_size;
//    emb_cpu->Resize(d);
//    const int64_t* ids_data = ids.data<int64_t>();
//    float* emb_cpu_data = emb_cpu->mutable_data<float>(cpu_place_);
//    const int numel = ids.numel();
//    for (int j = 0; j < numel; ++j) {
//      if (ids_data[j] == padding_idx_) {
//        memset(emb_cpu_data + emb_size * j, 0, emb_size * sizeof(float));
//        continue;
//      }
//      PADDLE_ENFORCE(ids_data[j] < voc_size, "Input id is out of the range of the vocabulary");
//      memcpy(emb_cpu_data + emb_size * j, table_data + emb_size * ids_data[j], emb_size * sizeof(float));
//    }
//  }
//}
//
//void ExecutorThreadWorker::LookupTableGrad(Scope* scope) {
//  LoDTensor* embedding = root_scope_->FindVar("embedding")->GetMutable<LoDTensor>();
//  float* table_data = embedding->mutable_data<float>(cpu_place_);
//  const size_t emb_size = embedding->dims()[1];
//  for (size_t i = 0; i < ids_names_.size(); ++i) {
//    LoDTensor* emb_grad_cpu = scope->FindVar(emb_grad_cpu_names_[i])->GetMutable<LoDTensor>();
//    float* emb_grad_cpu_data = emb_grad_cpu->mutable_data<float>(cpu_place_);
//
//    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
//    const int64_t* ids_data = ids.data<int64_t>();
//    const int numel = ids.numel();
//    for (int j = 0; j < numel; ++j) {
//      if (ids_data[j] == padding_idx_) {
//        continue;
//      }
//      // update embedding
//      memcpy(table_data + emb_size * ids_data[j], emb_grad_cpu_data + emb_size * j, emb_size * sizeof(float));
//    }
//  }
//}

void ExecutorThreadWorker::StartReaders() {
  for (auto& reader : readers_) {
    reader->Start();
  }
}

void ExecutorThreadWorker::StartEmbFFThreads() {
  emb_ff_stats_.resize(nemb_ff_threads_);
  reader_num_monitor_ = readers_.size();
  for (int i = 0; i < nemb_ff_threads_; ++i) {
    all_threads_.push_back(std::thread([this, i]() {
      int step_cnt = 0;
      int batch_size = 0;
      long accum_num = 0;
      Scope* scope = nullptr;
      DataFeed* reader = nullptr;

      platform::Timer timer;
      platform::Timer reader_timer;
      platform::Timer outer_timer;
      platform::Timer assign_scope_timer;
      bool started = false;
      while (scope_pool_->Receive(&scope)) {
        if (!started) {
          outer_timer.Start();
          started = true;
        }

        timer.Resume();
        if (!reader_queue_->Receive(&reader)) {
          break;
        }

        reader_timer.Resume();
        assign_scope_timer.Resume();
        reader->AssignFeedVar(*scope);
        assign_scope_timer.Pause();
        batch_size = reader->Next();
        reader_timer.Pause();

        if (batch_size <= 0) {
          std::lock_guard<std::mutex> lock(reader_num_mutex_);
          if (--reader_num_monitor_ <= 0) {
            reader_queue_->Close(); 
            break;
          }
          continue;
        }
        reader_queue_->Send(reader);
        accum_num += batch_size;

        LookupTableSumConcat(scope);

        emb_ff_scope_queue_->Send(scope);

        ++step_cnt;
        timer.Pause();
      }
      outer_timer.Pause();
      emb_ff_scope_queue_->Close();

      emb_ff_stats_[i].reader_ratio = reader_timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_ff_stats_[i].reader_us = reader_timer.ElapsedUS() / step_cnt;
      emb_ff_stats_[i].reader_throughput = accum_num / reader_timer.ElapsedSec();
      emb_ff_stats_[i].emb_ff_ratio = timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_ff_stats_[i].emb_ff_us = timer.ElapsedUS() / step_cnt;
      emb_ff_stats_[i].emb_ff_throughput = accum_num / outer_timer.ElapsedSec();
      emb_ff_stats_[i].other_ratio = assign_scope_timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_ff_stats_[i].other_us = assign_scope_timer.ElapsedUS() / step_cnt;
    }));
  }
}

template <typename T, int MajorType = Eigen::RowMajor, typename IndexType = Eigen::DenseIndex>
using EigenVector = EigenVector<T, MajorType, IndexType>;

void ExecutorThreadWorker::AsyncUpdateParam() {
  for (const std::string& param_name : param_names_) {
    LoDTensor* param_tensor = param_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
    LoDTensor* thread_tensor = thread_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
    auto param_tensor_e = EigenVector<float>::Flatten(*param_tensor);
    auto thread_tensor_e = EigenVector<float>::Flatten(*thread_tensor);
    if (false) param_tensor_e.device(*(gpu_dev_ctx_->eigen_device())) = thread_tensor_e - param_tensor_e;

    LoDTensor* delta_tensor = param_scope_->FindVar(param_name + "@DELTA")->GetMutable<LoDTensor>();
    cudaMemcpy(delta_tensor->mutable_data<float>(cpu_place_), param_tensor->data<float>(),
        param_tensor->numel() * sizeof(float), cudaMemcpyDeviceToHost);

    LoDTensor* root_tensor = root_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
    auto root_tensor_e = EigenVector<float>::Flatten(*root_tensor);
    auto delta_tensor_e = EigenVector<float>::Flatten(*delta_tensor);
    if (false) root_tensor_e.device(*(cpu_dev_ctx_->eigen_device())) = root_tensor_e + delta_tensor_e;

    cudaMemcpy(param_tensor->mutable_data<float>(gpu_place_), root_tensor->data<float>(),
        root_tensor->numel() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(thread_tensor->mutable_data<float>(gpu_place_), param_tensor->data<float>(),
        param_tensor->numel() * sizeof(float), cudaMemcpyDeviceToDevice);
  }
}

void ExecutorThreadWorker::StartGPUCalcThread() {
  all_threads_.push_back(std::thread([this]() {
    int step_cnt = 0;
    int accum_num = 0;
    Scope* scope = nullptr;
    platform::Timer outer_timer;
    platform::Timer main_timer;
    platform::Timer gpu_timer;
    platform::Timer memcpy_timer;
    bool started = false;
    while (emb_ff_scope_queue_->Receive(&scope)) {
      if (!started) {
        outer_timer.Start();
        started = true;
      }

      main_timer.Resume();
      memcpy_timer.Resume();
      const LoDTensor& user_concat_cpu = scope->FindVar("user_concat_cpu")->Get<LoDTensor>();
      const LoDTensor& news_concat_cpu = scope->FindVar("news_concat_cpu")->Get<LoDTensor>();
      const float* user_concat_cpu_data = user_concat_cpu.data<float>();
      const float* news_concat_cpu_data = news_concat_cpu.data<float>();
      LoDTensor* user_concat_gpu = scope->FindVar("user_concat")->GetMutable<LoDTensor>();
      LoDTensor* news_concat_gpu = scope->FindVar("news_concat")->GetMutable<LoDTensor>();
      user_concat_gpu->Resize(user_concat_cpu.dims());
      news_concat_gpu->Resize(news_concat_cpu.dims());
      float* user_concat_gpu_data = user_concat_gpu->mutable_data<float>(gpu_place_);
      float* news_concat_gpu_data = news_concat_gpu->mutable_data<float>(gpu_place_);
      cudaMemcpy(user_concat_gpu_data, user_concat_cpu_data, user_concat_cpu.memory_size(), cudaMemcpyHostToDevice);
      cudaMemcpy(news_concat_gpu_data, news_concat_cpu_data, news_concat_cpu.memory_size(), cudaMemcpyHostToDevice);
      accum_num += user_concat_cpu.dims()[0];
      memcpy_timer.Pause();

      gpu_timer.Resume();
      for (auto& op : ops_) {
        op->Run(*scope, gpu_place_);
      }
      gpu_timer.Pause();

      memcpy_timer.Resume();
      const LoDTensor& user_concat_grad_gpu = scope->FindVar("user_concat@GRAD")->Get<LoDTensor>();
      const LoDTensor& news_concat_grad_gpu = scope->FindVar("news_concat@GRAD")->Get<LoDTensor>();
      const float* user_concat_grad_gpu_data = user_concat_grad_gpu.data<float>();
      const float* news_concat_grad_gpu_data = news_concat_grad_gpu.data<float>();
      LoDTensor* user_concat_grad_cpu = scope->Var("user_concat_grad_cpu")->GetMutable<LoDTensor>();
      LoDTensor* news_concat_grad_cpu = scope->Var("news_concat_grad_cpu")->GetMutable<LoDTensor>();
      user_concat_grad_cpu->Resize(user_concat_grad_gpu.dims());
      news_concat_grad_cpu->Resize(news_concat_grad_gpu.dims());
      float* user_concat_grad_cpu_data = user_concat_grad_cpu->mutable_data<float>(cpu_place_);
      float* news_concat_grad_cpu_data = news_concat_grad_cpu->mutable_data<float>(cpu_place_);
      cudaMemcpy(user_concat_grad_cpu_data, user_concat_grad_gpu_data, user_concat_grad_gpu.memory_size(), cudaMemcpyDeviceToHost);
      cudaMemcpy(news_concat_grad_cpu_data, news_concat_grad_gpu_data, news_concat_grad_gpu.memory_size(), cudaMemcpyDeviceToHost);
      memcpy_timer.Pause();

      scope->DropKids();
      emb_bp_scope_queue_->Send(scope);

      if (++step_cnt % nasync_steps_ == 0) {
        AsyncUpdateParam();
      }

      main_timer.Pause();
    }

    outer_timer.Pause();
    emb_bp_scope_queue_->Close();

    main_net_stat_.memcpy_ratio = memcpy_timer.ElapsedSec() / outer_timer.ElapsedSec();
    main_net_stat_.memcpy_us = memcpy_timer.ElapsedUS() / step_cnt;
    main_net_stat_.gpu_ratio = gpu_timer.ElapsedSec() / outer_timer.ElapsedSec();
    main_net_stat_.gpu_us = gpu_timer.ElapsedUS() / step_cnt;
    main_net_stat_.main_net_ratio = main_timer.ElapsedSec() / outer_timer.ElapsedSec();
    main_net_stat_.main_net_us = main_timer.ElapsedUS() / step_cnt;
    main_net_stat_.other_ratio = main_net_stat_.main_net_ratio - main_net_stat_.memcpy_ratio - main_net_stat_.gpu_ratio;
    main_net_stat_.other_us = main_net_stat_.main_net_us - main_net_stat_.memcpy_us - main_net_stat_.gpu_us;
    main_net_stat_.throughput = accum_num / outer_timer.ElapsedSec();
  }));
}

void ExecutorThreadWorker::StartEmbBPThreads() {
  emb_bp_stats_.resize(nemb_bp_threads_);
  for (int i = 0; i < nemb_bp_threads_; ++i) {
    all_threads_.push_back(std::thread([this, i]() {
      int step_cnt = 0;
      long accum_num = 0;
      Scope* scope = nullptr;
      platform::Timer timer;
      platform::Timer outer_timer;
      bool started = false;
      while (emb_bp_scope_queue_->Receive(&scope)) {
        if (!started) {
          outer_timer.Start();
          started = true;
        }

        accum_num += scope->FindVar("user_concat_grad_cpu")->Get<LoDTensor>().dims()[0];

        timer.Resume();
        LookupTableSumConcatGrad(scope);
        timer.Pause();
        scope_pool_->Send(scope);

        ++step_cnt;
      }
      outer_timer.Pause();
      scope_pool_->Close();

      emb_bp_stats_[i].emb_bp_ratio = timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_bp_stats_[i].emb_bp_us = timer.ElapsedUS() / step_cnt;
      emb_bp_stats_[i].throughput = accum_num / outer_timer.ElapsedSec();
    }));
  }
}

void ExecutorThreadWorker::TrainFiles() {
  // TODO
  platform::SetNumThreads(1);

  int fetch_var_num = fetch_var_names_.size();
  fetch_values_.clear();
  fetch_values_.resize(fetch_var_num);

  StartReaders();
  StartEmbFFThreads();
  StartGPUCalcThread();
  StartEmbBPThreads();

  for (auto& t : all_threads_) {
    t.join();    
  }

  EmbFFStat emb_ff_stat;
  for (int i = 0; i < nemb_ff_threads_; ++i) {
    emb_ff_stat.reader_ratio += emb_ff_stats_[i].reader_ratio / nemb_ff_threads_;
    emb_ff_stat.reader_us += emb_ff_stats_[i].reader_us / nemb_ff_threads_;
    // no avg
    emb_ff_stat.reader_throughput += emb_ff_stats_[i].reader_throughput;

    emb_ff_stat.emb_ff_ratio += emb_ff_stats_[i].emb_ff_ratio / nemb_ff_threads_;
    emb_ff_stat.emb_ff_us += emb_ff_stats_[i].emb_ff_us / nemb_ff_threads_;
    // no avg
    emb_ff_stat.emb_ff_throughput += emb_ff_stats_[i].emb_ff_throughput;

    emb_ff_stat.other_ratio += emb_ff_stats_[i].other_ratio / nemb_ff_threads_;
    emb_ff_stat.other_us += emb_ff_stats_[i].other_us / nemb_ff_threads_;
  }
  fprintf(stderr, "emb_ff_perf reader:%.1f%%:%d emb_ff:%.1f%%:%d reader_trp:%d other:%.1f%%:%d\n",
      emb_ff_stat.reader_ratio * 100, static_cast<int>(emb_ff_stat.reader_us),
      emb_ff_stat.emb_ff_ratio * 100, static_cast<int>(emb_ff_stat.emb_ff_us),
      static_cast<int>(emb_ff_stat.reader_throughput),
      emb_ff_stat.other_ratio * 100, static_cast<int>(emb_ff_stat.other_us));

  EmbBPStat emb_bp_stat;
  for (int i = 0; i < nemb_bp_threads_; ++i) {
    emb_bp_stat.emb_bp_ratio += emb_bp_stats_[i].emb_bp_ratio / nemb_bp_threads_;
    emb_bp_stat.emb_bp_us += emb_bp_stats_[i].emb_bp_us / nemb_bp_threads_;
    emb_bp_stat.throughput += emb_bp_stats_[i].throughput;
  }
  fprintf(stderr, "emb_bp_perf emb_bp:%.1f%%:%d trp:%d\n",
      emb_bp_stat.emb_bp_ratio * 100, static_cast<int>(emb_bp_stat.emb_bp_us),
      static_cast<int>(emb_bp_stat.throughput));

  fprintf(stderr, "main_net_perf memcpy:%.1f%%:%d gpu:%.1f%%:%d other:%.1f%%:%d main_net:%.1f%%:%d\n",
      main_net_stat_.memcpy_ratio * 100, static_cast<int>(main_net_stat_.memcpy_us),
      main_net_stat_.gpu_ratio * 100, static_cast<int>(main_net_stat_.gpu_us),
      main_net_stat_.other_ratio * 100, static_cast<int>(main_net_stat_.other_us),
      main_net_stat_.main_net_ratio * 100, static_cast<int>(main_net_stat_.main_net_us));
  fprintf(stderr, "THROUGHPUT:%d\n", static_cast<int>(main_net_stat_.throughput));

  LOG(ERROR) << "Finish training";
}

}  // einit_modelnd namespace framework
}  // end namespace paddle
