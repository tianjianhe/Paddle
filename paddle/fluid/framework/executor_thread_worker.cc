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
#include "paddle/fluid/framework/async_executor.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/platform/timer.h"

#define EMB_SUM_CONCAT
#define PASGD

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

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,platform::dynload::ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void ExecutorThreadWorker::CreateThreadScope() {
  auto& block = main_program_->Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_, "root_scope should be set before creating thread scope");

  const std::vector<std::string>& input_feed = readers_[0]->GetUseSlotAlias();
  ids_names_.clear();
  for (auto& s : input_feed) {
    if (s != "click") {
      ids_names_.push_back(s);
    }
  }

  // the thread_scope reserves thread local parameters with memory in GPU
  thread_scope_.reset(new Scope);
  // the param_scope reserves global parameters for synchronization
  param_scope_.reset(new Scope);
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      // embedding parameter is not in main program
      const std::string var_name = var->Name();

      param_names_.push_back(var_name);

      auto& root_tensor = root_scope_->FindVar(var_name)->Get<LoDTensor>();

      auto* thread_tensor = thread_scope_->Var(var_name)->GetMutable<LoDTensor>();
      thread_tensor->Resize(root_tensor.dims());
      CUDACHECK(cudaMemcpy(thread_tensor->mutable_data<float>(gpu_place_),
          root_tensor.data<float>(),
          root_tensor.numel() * sizeof(float),
          cudaMemcpyHostToDevice));

      //auto* param_tensor = param_scope_->Var(var_name)->GetMutable<LoDTensor>();
      //param_tensor->Resize(root_tensor.dims());
      //CUDACHECK(cudaMemcpy(param_tensor->mutable_data<float>(gpu_place_),
      //    root_tensor.data<float>(),
      //    root_tensor.numel() * sizeof(float),
      //    cudaMemcpyHostToDevice));

      //// only used while synchronizing
      //auto* delta_tensor = param_scope_->Var(var_name + "@DELTA")->GetMutable<LoDTensor>();
      //delta_tensor->Resize(root_tensor.dims());
      //delta_tensor->mutable_data<float>(cpu_place_);
    }
  }

  // the scope is pipeline resource reserving computing context in flow cycle:
  // scope_pool_ -> emb_ff_scope_queue_ -> emb_bp_scope_queue_ -> scope_pool_
  scope_pool_.reset(new operators::reader::BlockingQueue<Scope*>(nscopes_));
  emb_ff_scope_queue_.reset(new operators::reader::BlockingQueue<Scope*>(nscopes_));
  emb_bp_scope_queue_.reset(new operators::reader::BlockingQueue<Scope*>(nscopes_));

  cudaSetDevice(rank_id_);
  for (int i = 0; i < nscopes_; ++i) {
    Scope* scope = &thread_scope_->NewScope();
    for (auto& var : block.AllVars()) {
      if (!var->Persistable()) {
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }

    cudaStream_t* copy_stream = scope->Var("copy_stream")->GetMutable<cudaStream_t>();
    CUDACHECK(cudaStreamCreate(copy_stream));

    scope_pool_->Send(scope);
  }

  //reader_queue_.reset(new operators::reader::BlockingQueue<DataFeed*>(readers_.size()));
  //for (auto& reader : readers_) {
  //  reader_queue_->Send(reader.get());
  //}
  PADDLE_ENFORCE(readers_.size() == (size_t) nemb_ff_threads_, "readers_.size() != emb_ff_threads_");
}

void ExecutorThreadWorker::CreateThreadOperators() {
  auto& block = main_program_->Block(0);
  op_names_.clear();
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    // Do we need to call "delete" in destruction?
    OperatorBase* local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
  }
}

void ExecutorThreadWorker::CreateThreadResource() {
  CreateThreadScope();
  CreateThreadOperators();
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

void ExecutorThreadWorker::LookupTable(Scope* scope) {
  const LoDTensor& embedding = root_scope_->FindVar("embedding")->Get<LoDTensor>();
  const float* table_data = embedding.data<float>();
  const int voc_size = embedding.dims()[0];
  const int emb_size = embedding.dims()[1];

  for (size_t i = 0; i < ids_names_.size(); ++i) {
    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
    LoDTensor* emb_cpu = scope->Var(ids_names_[i] + "_emb_cpu")->GetMutable<LoDTensor>();
    emb_cpu->set_lod(ids.lod());
    DDim d = ids.dims();
    d[1] = emb_size;
    emb_cpu->Resize(d);
    const uint64_t* ids_data = reinterpret_cast<const uint64_t*>(ids.data<int64_t>());
    float* emb_cpu_data = emb_cpu->mutable_data<float>(pinned_place_);
    const int numel = ids.numel();
    for (int j = 0; j < numel; ++j) {
      if (ids_data[j] == padding_idx_) {
        memset(emb_cpu_data + emb_size * j, 0, emb_size * sizeof(float));
        continue;
      }

      uint64_t mod_id = ids_data[j] % voc_size;
      memcpy(emb_cpu_data + emb_size * j, table_data + emb_size * mod_id,
          emb_size * sizeof(float));
    }

    LoDTensor* emb_gpu = scope->FindVar(ids_names_[i] + "_emb")->GetMutable<LoDTensor>();
    emb_gpu->set_lod(emb_cpu->lod());
    emb_gpu->Resize(emb_cpu->dims());
    float* emb_gpu_data = emb_gpu->mutable_data<float>(gpu_place_);
    cudaStream_t* copy_stream = scope->FindVar("copy_stream")->GetMutable<cudaStream_t>();
    CUDACHECK(cudaMemcpyAsync(emb_gpu_data, emb_cpu_data, emb_cpu->memory_size(),
        cudaMemcpyHostToDevice, *copy_stream));
  }
}

void ExecutorThreadWorker::LookupTableGrad(Scope* scope) {
  LoDTensor* embedding = root_scope_->FindVar("embedding")->GetMutable<LoDTensor>();
  float* table_data = embedding->mutable_data<float>(cpu_place_);
  const size_t voc_size = embedding->dims()[0];
  const int emb_size = embedding->dims()[1];
  //TODO
  //float lr = 0.01;
  float lr = root_scope_->FindVar("learning_rate_0")->Get<LoDTensor>().data<float>()[0];

  for (size_t i = 0; i < ids_names_.size(); ++i) {
    LoDTensor* emb_grad_cpu = scope->Var(ids_names_[i] + "_emb_grad_cpu")->GetMutable<LoDTensor>();
    float* emb_grad_cpu_data = emb_grad_cpu->mutable_data<float>(pinned_place_);

    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
    const uint64_t* ids_data = reinterpret_cast<const uint64_t*>(ids.data<int64_t>());
    const int numel = ids.numel();
    for (int j = 0; j < numel; ++j) {
      if (ids_data[j] == padding_idx_) {
        continue;
      }
      uint64_t mod_id = ids_data[j] % voc_size;
      cpu_blas_->AXPY(emb_size, -lr, emb_grad_cpu_data + emb_size * j,
          table_data + emb_size * mod_id);
    }
  }
}

// OPTIMIZE: hard coding
static const int user_slot_num = 73;
static const int news_slot_num = 44;

void ExecutorThreadWorker::LookupTableSumConcat(Scope* scope) {
  const LoDTensor& embedding = root_scope_->FindVar("embedding")->Get<LoDTensor>();
  const float* table_data = embedding.data<float>();
  const int voc_size = embedding.dims()[0];
  const int emb_size = embedding.dims()[1];

  const int batch_size = scope->FindVar(ids_names_[0])->Get<LoDTensor>().lod()[0].size() - 1;
  PADDLE_ENFORCE(ids_names_.size() == user_slot_num + news_slot_num, "mismatch");

  LoDTensor* user_concat_cpu = scope->Var("user_concat_cpu")->GetMutable<LoDTensor>();
  LoDTensor* news_concat_cpu = scope->Var("news_concat_cpu")->GetMutable<LoDTensor>();
  user_concat_cpu->Resize({batch_size, user_slot_num * emb_size});
  news_concat_cpu->Resize({batch_size, news_slot_num * emb_size});
  float* user_concat_cpu_data = user_concat_cpu->mutable_data<float>(pinned_place_);
  float* news_concat_cpu_data = news_concat_cpu->mutable_data<float>(pinned_place_);
  memset(user_concat_cpu_data, 0, user_concat_cpu->memory_size());
  memset(news_concat_cpu_data, 0, news_concat_cpu->memory_size());

  float* concat_data = user_concat_cpu_data;
  int concat_size = emb_size * user_slot_num;
  for (size_t i = 0; i < ids_names_.size(); ++i) {
    if (i == user_slot_num) {
      concat_data = news_concat_cpu_data;
      concat_size = emb_size * news_slot_num;
    }
    const int slot_idx = i % user_slot_num;

    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
    const uint64_t* ids_data = reinterpret_cast<const uint64_t*>(ids.data<int64_t>());
    const auto& lod = ids.lod()[0];
    for (int bs = 0; bs < batch_size; ++bs) {
      int end = lod[bs + 1];
      for (int offset = lod[bs]; offset < end; ++offset) {
        if (ids_data[offset] == padding_idx_) {
          continue;
        }

        uint64_t mod_id = ids_data[offset] % voc_size;
        cpu_blas_->VADD(emb_size, table_data + emb_size * mod_id,
            concat_data + concat_size * bs + emb_size * slot_idx,
            concat_data + concat_size * bs + emb_size * slot_idx);
      }
    }
  }

  LoDTensor* user_concat_gpu = scope->FindVar("user_concat")->GetMutable<LoDTensor>();
  LoDTensor* news_concat_gpu = scope->FindVar("news_concat")->GetMutable<LoDTensor>();
  user_concat_gpu->Resize(user_concat_cpu->dims());
  news_concat_gpu->Resize(news_concat_cpu->dims());
  float* user_concat_gpu_data = user_concat_gpu->mutable_data<float>(gpu_place_);
  float* news_concat_gpu_data = news_concat_gpu->mutable_data<float>(gpu_place_);

  const LoDTensor& click_cpu = scope->FindVar("click")->Get<LoDTensor>();
  const int64_t* click_cpu_data = reinterpret_cast<const int64_t*>(click_cpu.data<int64_t>());
  int64_t *tmp = new int64_t[batch_size];
  memcpy(tmp, click_cpu_data, batch_size * sizeof(int64_t));
  LoDTensor* click_gpu = scope->FindVar("click")->GetMutable<LoDTensor>();
  int64_t* click_gpu_data = click_gpu->mutable_data<int64_t>(gpu_place_);



  cudaStream_t* copy_stream = scope->FindVar("copy_stream")->GetMutable<cudaStream_t>();
  CUDACHECK(cudaMemcpyAsync(user_concat_gpu_data, user_concat_cpu_data,
        user_concat_cpu->memory_size(), cudaMemcpyHostToDevice, *copy_stream));
  CUDACHECK(cudaMemcpyAsync(news_concat_gpu_data, news_concat_cpu_data,
        news_concat_cpu->memory_size(), cudaMemcpyHostToDevice, *copy_stream));
  CUDACHECK(cudaMemcpyAsync(click_gpu_data, tmp,
        batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice, *copy_stream));
  delete []tmp;
}

void ExecutorThreadWorker::LookupTableSumConcatGrad(Scope* scope) {
  LoDTensor* embedding = root_scope_->FindVar("embedding")->GetMutable<LoDTensor>();
  float* table_data = embedding->mutable_data<float>(cpu_place_);
  const size_t voc_size = embedding->dims()[0];
  const size_t emb_size = embedding->dims()[1];
  const int batch_size = scope->FindVar(ids_names_[0])->Get<LoDTensor>().lod()[0].size() - 1;
  //TODO
  //float lr = 0.01;
  float lr = *(root_scope_->FindVar("learning_rate_0")->Get<LoDTensor>().data<float>());

  float* user_concat_grad_cpu_data =
    scope->Var("user_concat_grad_cpu")->GetMutable<LoDTensor>()->mutable_data<float>(pinned_place_);
  float* news_concat_grad_cpu_data =
    scope->Var("news_concat_grad_cpu")->GetMutable<LoDTensor>()->mutable_data<float>(pinned_place_);

  const float* concat_grad_cpu_data = user_concat_grad_cpu_data;
  int concat_size = emb_size * user_slot_num;
  for (size_t i = 0; i < ids_names_.size(); ++i) {
    if (i == user_slot_num) {
      concat_grad_cpu_data = news_concat_grad_cpu_data;
      concat_size = emb_size * news_slot_num;
    }
    const int slot_idx = i % user_slot_num;

    const LoDTensor& ids = scope->FindVar(ids_names_[i])->Get<LoDTensor>();
    const uint64_t* ids_data = reinterpret_cast<const uint64_t*>(ids.data<int64_t>());
    const auto& lod = ids.lod()[0];
    for (int bs = 0; bs < batch_size; ++bs) {
      int end = lod[bs + 1];
      for (int offset = lod[bs]; offset < end; ++offset) {
        if (ids_data[offset] == padding_idx_) {
          continue;
        }

        uint64_t mod_id = ids_data[offset] % voc_size;
        cpu_blas_->AXPY(emb_size, -lr, concat_grad_cpu_data + concat_size * bs + emb_size * slot_idx,
            table_data + emb_size * mod_id);
      }
    }
  }
}

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
      long step_cnt = 0;
      int batch_size = 0;
      long accum_num = 0;
      Scope* scope = nullptr;
      DataFeed* reader = nullptr;

      platform::Timer emb_ff_timer;
      platform::Timer reader_timer;
      platform::Timer outer_timer;
      bool started = false;
      while (scope_pool_->Receive(&scope)) {
        if (!started) {
          outer_timer.Start();
          started = true;
        }

        emb_ff_timer.Resume();
        //if (!reader_queue_->Receive(&reader)) {
        //  break;
        //}
        reader = readers_[i].get();

        reader_timer.Resume();
        reader->AssignFeedVar(*scope);
        batch_size = reader->Next();
        reader_timer.Pause();
        if (batch_size <= 0) {
          break;
        }
        //reader_queue_->Send(reader);

#ifdef EMB_SUM_CONCAT
        LookupTableSumConcat(scope);
#else
        LookupTable(scope);
#endif

        emb_ff_scope_queue_->Send(scope);

        ++step_cnt;
        accum_num += batch_size;

        emb_ff_timer.Pause();
      }
      outer_timer.Pause();

      {
        std::lock_guard<std::mutex> lock(reader_num_mutex_);
        if (--reader_num_monitor_ <= 0) {
          //reader_queue_->Close();
          while (emb_ff_scope_queue_->Size()) {
            sleep(1);
          }
          emb_ff_scope_queue_->Close();
        }
      }

      LOG(ERROR) << "r" << rank_id_ << "t" << i << "_emb_ff_steps: " << step_cnt;

      emb_ff_stats_[i].reader_ratio = reader_timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_ff_stats_[i].reader_us = reader_timer.ElapsedUS() / step_cnt;
      emb_ff_stats_[i].reader_throughput = accum_num / reader_timer.ElapsedSec();
      emb_ff_stats_[i].emb_ff_ratio = emb_ff_timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_ff_stats_[i].emb_ff_us = emb_ff_timer.ElapsedUS() / step_cnt;
      emb_ff_stats_[i].throughput = accum_num / outer_timer.ElapsedSec();
    }));
  }
}

void ExecutorThreadWorker::StartEmbBPThreads() {
  emb_bp_stats_.resize(nemb_bp_threads_);
  for (int i = 0; i < nemb_bp_threads_; ++i) {
    all_threads_.push_back(std::thread([this, i]() {
      long step_cnt = 0;
      long accum_num = 0;
      Scope* scope = nullptr;
      platform::Timer emb_bp_timer;
      platform::Timer wait_timer;
      platform::Timer outer_timer;
      bool started = false;
      while (emb_bp_scope_queue_->Receive(&scope)) {
        if (!started) {
          outer_timer.Start();
          started = true;
        }

        emb_bp_timer.Resume();

        wait_timer.Resume();
        // wait util GPU2CPU memcpy finishs for further CPU calc
        cudaStream_t* copy_stream = scope->FindVar("copy_stream")->GetMutable<cudaStream_t>();
        CUDACHECK(cudaStreamSynchronize(*copy_stream));
        wait_timer.Pause();

#ifdef EMB_SUM_CONCAT
        LookupTableSumConcatGrad(scope);
#else
        LookupTableGrad(scope);
#endif

        //if (step_cnt % 1000 == 0) {
        if (step_cnt >= 0) {
          LogFetchValues(*scope);
        }

        scope_pool_->Send(scope);
        ++step_cnt;
        accum_num += scope->FindVar(ids_names_[0])->Get<LoDTensor>().dims()[0];

        emb_bp_timer.Pause();
      }
      outer_timer.Pause();
      scope_pool_->Close();

      LOG(ERROR) << "r" << rank_id_ << "t" << i << "_emb_bp_steps: " << step_cnt;

      emb_bp_stats_[i].emb_bp_ratio = emb_bp_timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_bp_stats_[i].emb_bp_us = emb_bp_timer.ElapsedUS() / step_cnt;
      emb_bp_stats_[i].wait_ratio = wait_timer.ElapsedSec() / outer_timer.ElapsedSec();
      emb_bp_stats_[i].wait_us = wait_timer.ElapsedUS() / step_cnt;
      emb_bp_stats_[i].throughput = accum_num / emb_bp_timer.ElapsedSec();
    }));
  }
}

//void ExecutorThreadWorker::AsyncUpdateParam() {
//  for (const std::string& param_name : param_names_) {
//    LoDTensor* param_tensor = param_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
//    LoDTensor* thread_tensor = thread_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
//    float* param_data = param_tensor->mutable_data<float>(gpu_place_);
//    float* thread_data = thread_tensor->mutable_data<float>(gpu_place_);
//    const int numel = thread_tensor->numel();
//    gpu_blas_->AXPY(numel, -1, thread_data, param_data);
//
//    LoDTensor* delta_tensor = param_scope_->FindVar(param_name + "@DELTA")->GetMutable<LoDTensor>();
//    float* delta_data = delta_tensor->mutable_data<float>(cpu_place_);
//    cudaMemcpy(delta_data, param_data, numel * sizeof(float), cudaMemcpyDeviceToHost);
//
//    LoDTensor* root_tensor = root_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
//    float* root_data = root_tensor->mutable_data<float>(cpu_place_);
//    cpu_blas_->AXPY(numel, -1, delta_data, root_data);
//
//    cudaMemcpy(param_data, root_data, numel * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(thread_data, param_data, numel * sizeof(float), cudaMemcpyDeviceToDevice);
//  }
//}

void ExecutorThreadWorker::SyncParam() {
  if (nranks_ == 1) {
    return;
  }

  // OPTIMIZE: merge pameters into a continuous memory place
  for (const std::string& param_name : param_names_) {
    LoDTensor* thread_tensor = thread_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
    float* thread_data = thread_tensor->mutable_data<float>(gpu_place_);
    const int numel = thread_tensor->numel();

    NCCLCHECK(platform::dynload::ncclAllReduce(thread_data, thread_data, numel,
          ncclFloat, ncclSum, *nccl_comm_, gpu_dev_ctx_->stream()));
  }
  //CUDACHECK(cudaStreamSynchronize(gpu_dev_ctx_->stream()));
  gpu_dev_ctx_->Wait();

  // TODO: scale params
  //for (const std::string& param_name : param_names_) {
  //  LoDTensor* thread_tensor = thread_scope_->FindVar(param_name)->GetMutable<LoDTensor>();
  //  float* thread_data = thread_tensor->mutable_data<float>(gpu_place_);
  //  const int numel = thread_tensor->numel();
  //  gpu_blas_->SCAL(numel, 1.0 / nranks_, thread_data);
  //}
}

void ExecutorThreadWorker::StartGPUCalc() {
  int periodic_cnt = 0;
  long step_cnt = 0;
  long accum_num = 0;
  Scope* scope = nullptr;
  platform::Timer outer_timer;
  platform::Timer main_timer;
  platform::Timer gpu_timer;
  platform::Timer wait_timer;
  platform::Timer sync_timer;

  bool started = false;
  while (emb_ff_scope_queue_->Receive(&scope)) {
    if (!started) {
      outer_timer.Start();
      started = true;
    }

    main_timer.Resume();

    wait_timer.Resume();
    // wait for CPU2GPU memcpy finishing, as the cudaMemcpy and GPU calc are in different streams
    CUDACHECK(cudaStreamSynchronize(*(scope->FindVar("copy_stream")->GetMutable<cudaStream_t>())));
    wait_timer.Pause();

    gpu_timer.Resume();
    for (auto& op : ops_) {
      op->Run(*scope, gpu_place_);
    }
    gpu_timer.Pause();

    wait_timer.Resume();
    // wait for GPU calc finising, as the cudaMemcpy and GPU calc are in different streams
    gpu_dev_ctx_->Wait();
    wait_timer.Pause();

    scope->DropKids();

    // asynchronously start GPU2CPU memcpy
#ifdef EMB_SUM_CONCAT
    const LoDTensor& user_concat_grad_gpu = scope->FindVar("user_concat@GRAD")->Get<LoDTensor>();
    const LoDTensor& news_concat_grad_gpu = scope->FindVar("news_concat@GRAD")->Get<LoDTensor>();
    const float* user_concat_grad_gpu_data = user_concat_grad_gpu.data<float>();
    const float* news_concat_grad_gpu_data = news_concat_grad_gpu.data<float>();
    LoDTensor* user_concat_grad_cpu = scope->Var("user_concat_grad_cpu")->GetMutable<LoDTensor>();
    LoDTensor* news_concat_grad_cpu = scope->Var("news_concat_grad_cpu")->GetMutable<LoDTensor>();
    user_concat_grad_cpu->Resize(user_concat_grad_gpu.dims());
    news_concat_grad_cpu->Resize(news_concat_grad_gpu.dims());
    float* user_concat_grad_cpu_data = user_concat_grad_cpu->mutable_data<float>(pinned_place_);
    float* news_concat_grad_cpu_data = news_concat_grad_cpu->mutable_data<float>(pinned_place_);
    cudaStream_t* copy_stream = scope->FindVar("copy_stream")->GetMutable<cudaStream_t>();
    wait_timer.Resume();
    CUDACHECK(cudaMemcpyAsync(user_concat_grad_cpu_data, user_concat_grad_gpu_data,
          user_concat_grad_gpu.memory_size(), cudaMemcpyDeviceToHost, *copy_stream));
    CUDACHECK(cudaMemcpyAsync(news_concat_grad_cpu_data, news_concat_grad_gpu_data,
          news_concat_grad_gpu.memory_size(), cudaMemcpyDeviceToHost, *copy_stream));
    wait_timer.Pause();
#else
    for (size_t i = 0; i < ids_names_.size(); ++i) {
      const LoDTensor& emb_grad_gpu = scope->FindVar(ids_names_[i] + "_emb@GRAD")->Get<LoDTensor>();
      const float* emb_grad_gpu_data = emb_grad_gpu.data<float>();
      LoDTensor* emb_grad_cpu = scope->Var(ids_names_[i] + "_emb_grad_cpu")->GetMutable<LoDTensor>();
      emb_grad_cpu->Resize(emb_grad_gpu.dims());
      float* emb_grad_cpu_data = emb_grad_cpu->mutable_data<float>(pinned_place_);
      cudaStream_t* copy_stream = scope->FindVar("copy_stream")->GetMutable<cudaStream_t>();
      wait_timer.Resume();
      CUDACHECK(cudaMemcpyAsync(emb_grad_cpu_data, emb_grad_gpu_data, emb_grad_gpu.memory_size(),
            cudaMemcpyDeviceToHost, *copy_stream));
    }
#endif

    emb_bp_scope_queue_->Send(scope);

    if (++periodic_cnt % nasync_steps_ == 0) {
      exe_->UpdateSyncFlag(rank_id_);
    }

    if (sync_signal_) {
      sync_timer.Resume();
      SyncParam();
      sync_signal_ = false;
      periodic_cnt = 0;
      sync_timer.Pause();
    }

    ++step_cnt;
    accum_num += scope->FindVar(ids_names_[0])->Get<LoDTensor>().dims()[0];
    main_timer.Pause();
  }

  outer_timer.Pause();

  while (emb_bp_scope_queue_->Size()) {
    sleep(1);
  }
  emb_bp_scope_queue_->Close();

  main_net_stat_.gpu_ratio = gpu_timer.ElapsedSec() / outer_timer.ElapsedSec();
  main_net_stat_.gpu_us = gpu_timer.ElapsedUS() / step_cnt;
  main_net_stat_.gpu_trp = accum_num / gpu_timer.ElapsedSec();
  main_net_stat_.main_net_ratio = main_timer.ElapsedSec() / outer_timer.ElapsedSec();
  main_net_stat_.main_net_us = main_timer.ElapsedUS() / step_cnt;
  main_net_stat_.main_net_trp = accum_num / main_timer.ElapsedSec();
  main_net_stat_.other_ratio = main_net_stat_.main_net_ratio - main_net_stat_.gpu_ratio;
  main_net_stat_.other_us = main_net_stat_.main_net_us - main_net_stat_.gpu_us;
  main_net_stat_.wait_ratio = wait_timer.ElapsedSec() / outer_timer.ElapsedSec();
  main_net_stat_.wait_us = wait_timer.ElapsedUS() / wait_timer.Count();
  main_net_stat_.sync_ratio = sync_timer.Count() ? (sync_timer.ElapsedSec() / outer_timer.ElapsedSec()) : 0;
  main_net_stat_.sync_us = sync_timer.Count() ? (sync_timer.ElapsedUS() / sync_timer.Count()) : 0;
  main_net_stat_.throughput = accum_num / outer_timer.ElapsedSec();
}

void ExecutorThreadWorker::LogFetchValues(Scope& scope) {
  // TODO: inspect loss temporarily
  for (auto& name : fetch_var_names_) {
    const LoDTensor &loss_gpu = scope.FindVar(name)->Get<LoDTensor>();
    const float* loss_gpu_data = loss_gpu.data<float>();

    LoDTensor* loss_cpu = scope.Var(name+"cpu")->GetMutable<LoDTensor>();
    loss_cpu->Resize(loss_gpu.dims());
    float *loss_cpu_data = loss_cpu->mutable_data<float>(cpu_place_);
    cudaMemcpy(loss_cpu_data, loss_gpu_data, loss_gpu.dims()[0]*sizeof(float), cudaMemcpyDeviceToHost);

    //For infer temporary
    //Note: fprintf may not be thread-safe
    if (name.find("sigmoid") != std::string::npos){
        FILE *out = fopen("pred.out", "a");
        for (int i = 0; i<loss_gpu.dims()[0]; ++i)
          fprintf(out, "%.4lf\n", loss_cpu_data[i]);
        fclose(out);
    }
    if (name.find("cast") != std::string::npos){
        FILE *out = fopen("label.out", "a");
        for (int i = 0; i<loss_gpu.dims()[0]; ++i)
          if (loss_cpu_data[i] > 0.5)
            fprintf(out, "1\n");
          else
            fprintf(out,"0\n");
        fclose(out);
    }
  }
}

void ExecutorThreadWorker::TrainFiles() {
  // TODO
  platform::SetNumThreads(1);
  CUDACHECK(cudaSetDevice(rank_id_));

  fetch_values_.clear();
  fetch_values_.resize(fetch_var_names_.size());

  StartReaders();
  StartEmbFFThreads();
  StartEmbBPThreads();
  StartGPUCalc();

  for (auto& t : all_threads_) {
    t.join();    
  }

  auto& block = main_program_->Block(0);
  for (auto& var : block.AllVars()) {
    if (!var->Persistable()) continue;

    const std::string var_name = var->Name();
    auto *root_tensor = root_scope_->Var(var_name)->GetMutable<LoDTensor>();
    auto& thread_tensor = thread_scope_->FindVar(var_name)->Get<LoDTensor>();
    CUDACHECK(cudaMemcpy(root_tensor->mutable_data<float>(cpu_place_),
      thread_tensor.data<float>(),
      thread_tensor.numel() * sizeof(float),
      cudaMemcpyDeviceToHost));
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
    emb_ff_stat.throughput += emb_ff_stats_[i].throughput;
  }
  fprintf(stderr, "r%d_emb_ff_perf reader:%.1f%%:%d emb_ff:%.1f%%:%d reader_trp:%d\n", rank_id_,
      emb_ff_stat.reader_ratio * 100, static_cast<int>(emb_ff_stat.reader_us),
      emb_ff_stat.emb_ff_ratio * 100, static_cast<int>(emb_ff_stat.emb_ff_us),
      static_cast<int>(emb_ff_stat.reader_throughput));

  EmbBPStat emb_bp_stat;
  for (int i = 0; i < nemb_bp_threads_; ++i) {
    emb_bp_stat.emb_bp_ratio += emb_bp_stats_[i].emb_bp_ratio / nemb_bp_threads_;
    emb_bp_stat.emb_bp_us += emb_bp_stats_[i].emb_bp_us / nemb_bp_threads_;
    emb_bp_stat.wait_ratio += emb_bp_stats_[i].wait_ratio / nemb_bp_threads_;
    emb_bp_stat.wait_us += emb_bp_stats_[i].wait_us / nemb_bp_threads_;
    emb_bp_stat.throughput += emb_bp_stats_[i].throughput;
  }
  fprintf(stderr, "r%d_emb_bp_perf emb_bp:%.1f%%:%d wait:%.1f%%:%d trp:%d\n", rank_id_,
      emb_bp_stat.emb_bp_ratio * 100, static_cast<int>(emb_bp_stat.emb_bp_us),
      emb_bp_stat.wait_ratio * 100, static_cast<int>(emb_bp_stat.wait_us),
      static_cast<int>(emb_bp_stat.throughput));

  fprintf(stderr, "r%d_main_net_perf gpu:%.1f%%:%d wait:%.1f%%:%d sync:%.1f%%:%d main_net:%.1f%%:%d\n", rank_id_,
      main_net_stat_.gpu_ratio * 100, static_cast<int>(main_net_stat_.gpu_us),
      main_net_stat_.wait_ratio * 100, static_cast<int>(main_net_stat_.wait_us),
      main_net_stat_.sync_ratio * 100, static_cast<int>(main_net_stat_.sync_us),
      main_net_stat_.main_net_ratio * 100, static_cast<int>(main_net_stat_.main_net_us));
  fprintf(stderr, "r%d_THROUGHPUT:%d\n", rank_id_, static_cast<int>(main_net_stat_.throughput));

  //fprintf(stderr, "r%d_other_perf gpu_trp:%d main_net_trp:%d wait:%.1f%%:%.1f\n", rank_id_,
  //    static_cast<int>(main_net_stat_.gpu_trp),
  //    static_cast<int>(main_net_stat_.main_net_trp), main_net_stat_.wait_ratio, main_net_stat_.wait_us);

  for (Scope* scope : thread_scope_->kids()) {
    cudaStream_t* copy_stream = scope->FindVar("copy_stream")->GetMutable<cudaStream_t>();
    CUDACHECK(cudaStreamDestroy(*copy_stream));
  }
  thread_scope_->DropKids();

  LOG(ERROR) << "r" << rank_id_ << ": Finish training";
}

}  // einit_modelnd namespace framework
}  // end namespace paddle
