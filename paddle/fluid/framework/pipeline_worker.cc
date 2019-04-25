/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"
#include "paddle/fluid/platform/device_context.h"

//#define DEBUG
//#define PIPELINE_PROFILE

#ifdef PIPELINE_PROFILE

#define PIPELINE_PROFILE_DECLARE(type) platform::Timer type##_timer
#define PIPELINE_PROFILE_START(type) type##_timer.Start()
#define PIPELINE_PROFILE_RESUME(type) type##_timer.Resume()
#define PIPELINE_PROFILE_PAUSE(type) type##_timer.Pause()

#else

#define PIPELINE_PROFILE_DECLARE(type)
#define PIPELINE_PROFILE_START(type)
#define PIPELINE_PROFILE_RESUME(type)
#define PIPELINE_PROFILE_PAUSE(type)
#define PIPELINE_PROFILE_STOP(type)

#endif

namespace paddle {
namespace framework {

std::unique_ptr<platform::NCCLContextMap> SynchronizeFunctor::nccl_ctx_map_ = nullptr;
std::vector<Scope*>	SynchronizeFunctor::pipeline_scopes_;
//std::unique_ptr<ncclUniqueId> SynchronizeFunctor::nccl_id_ = nullptr;
//
uint64_t SynchronizeFunctor::s_sync_flag_ = 0;

int ReadFunctor::operator()(Scope* scope, int index) {
  // TODO: readers in a queue need to be compared
  if (index >= readers_.size()) {
    return -1;
  }
  readers_[index]->AssignFeedVar(*scope);
  return readers_[index]->Next();
}

SynchronizeFunctor::SynchronizeFunctor(int rank_id, int rank_num, int nasync_steps)
    : kRankId(rank_id), kNRanks(rank_num), kNAsyncSteps(nasync_steps) {
  PADDLE_ENFORCE(rank_num > 1, "rank_num %d should larger than 1");

  if (!nccl_ctx_map_) {
    static std::mutex mutex;
    mutex.lock();
    if (!nccl_ctx_map_) {
      //nccl_id_.reset(new ncclUniqueId);
      //PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(nccl_id.get()));
      std::vector<platform::Place> cuda_places;
      for (int i = 0; i < kNRanks; ++i) {
        cuda_places.emplace_back(platform::CUDAPlace(i));
      }
      //nccl_ctx_map_.reset(new platform::NCCLContextMap(cuda_places, nccl_id_.get(), nranks_, rank_id_));
      nccl_ctx_map_.reset(new platform::NCCLContextMap(cuda_places));
    }
    mutex.unlock();
  }
  if (pipeline_scopes_.empty()) {
    static std::mutex pmutex;
    pmutex.lock();
    if (pipeline_scopes_.empty()) {
      pipeline_scopes_.resize(kNRanks);
    }
    pmutex.unlock();
  }

  counter_ = 0;
  sync_signal_ = 0;
  uint8_t* ptr = reinterpret_cast<uint8_t*>(&sync_signal_);
  for (int i = 0; i < kNRanks; ++i) {
    ptr[i] = 0xFF;
  }
}

int SynchronizeFunctor::operator()(Scope* scope, int index) {
  ++counter_;
  if (counter_ < kNAsyncSteps) {
    return 0;
  }

  printf("htj in SynchronizeFunctor: %d, rankid: %d\n", __LINE__, kRankId);
  if (counter_ == kNAsyncSteps) {
    reinterpret_cast<uint8_t*>(&s_sync_flag_)[kRankId] = 0xFF;
  }

  printf("htj in SynchronizeFunctor: %d, rankid: %d\n", __LINE__, kRankId);
  if (s_sync_flag_ == sync_signal_) {
    static std::mutex mutex;
    if (mutex.try_lock()) {
      if (s_sync_flag_ == sync_signal_) {
        Synchronize();
        s_sync_flag_ = 0;
      }
      mutex.unlock();
    }
  }
  printf("htj in SynchronizeFunctor: %d, rankid: %d\n", __LINE__, kRankId);

  if (s_sync_flag_ == 0) {
    counter_ = 0;
  }

  return 0;
}

void SynchronizeFunctor::Synchronize() {
  printf("htj in Synchronize\n");
  for (const std::string& name : sync_param_names_) {
      printf("htj in Synchronize: var name: %s\n", name.c_str());
    platform::NCCLGroupGuard guard;
    for (int i = 0; i < kNRanks; ++i) {
      //platform::NCCLGroupGuard guard;
      const platform::NCCLContext& nccl_ctx = nccl_ctx_map_->at(i);
      LoDTensor* tensor = pipeline_scopes_[i]->FindVar(name)->GetMutable<LoDTensor>();
      // FIXME: do not depend on data type explicitly
      float* data = tensor->mutable_data<float>(nccl_ctx_map_->DevCtx(i)->GetPlace());
      const int numel = tensor->numel();
      printf("htj in Synchronize: before all reduce in rank %d\n", i);
      PADDLE_ENFORCE(platform::dynload::ncclAllReduce(data, data, numel,
            ncclFloat, ncclSum, nccl_ctx.comm(), nccl_ctx.stream()), "nccl all reduce error");
      LOG(ERROR) << "Sync " << name;
      // TODO: average reduced param
    }
  }
  nccl_ctx_map_->WaitAll();
}

//#define SEC_V3LOG VLOG(3) << "[r" << kRankId << "s" << sec_index_ << "t" << i << "] "
#define SEC_V3LOG LOG(ERROR) << "[r" << kRankId << "s" << sec_index_ << "t" << i << "] "

PipeSection::PipeSection(const PipeSectionConfig& cfg, int rank_id) : kRankId(rank_id) {
  program_.reset(new ProgramDesc(cfg.program_desc()));
  read_func_ = nullptr;
  sync_func_ = nullptr;
  pre_ = nullptr;
  next_ = nullptr;

  switch (cfg.place())  {
    case PipeSectionConfig::CPUPlace:
      place_ = platform::CPUPlace();
      break;

    case PipeSectionConfig::CUDAPlace:
      place_ = platform::CUDAPlace(kRankId);
      break;

    case PipeSectionConfig::CUDAPinnedPlace:
      place_ = platform::CUDAPinnedPlace();
      break;

    default:
      PADDLE_ENFORCE(false, "Unkown place type in PipeSectionConfig: %d", cfg.place());
  }

  concurrency_ = cfg.concurrency();
  joint_in_var_names_.assign(cfg.joint_in_var_names().begin(), cfg.joint_in_var_names().end());
  joint_out_var_names_.assign(cfg.joint_out_var_names().begin(), cfg.joint_out_var_names().end());

  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);

  param_names_.clear();
  for (auto& var : program_->Block(0).AllVars()) {
    if (var->Persistable()) {
      param_names_.push_back(var->Name());
    }
  }
}

void PipeSection::CreateOperators() {
  for (auto& op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
}

void PipeSection::CopyParameters(const Scope& root_scope, Scope* pipeline_scope) {
  if (!platform::is_gpu_place(place_)) {
    // Hogwild training strategy is applied to pipe sections in CPU or CUDAPinned place,
    // with parameters reserved in root_scope.
    return;
  }

  for (const std::string& name : param_names_) {
    const LoDTensor& root_tensor = root_scope.FindVar(name)->Get<LoDTensor>();

		// TODO: check a new var of the same name is created in pipeline_scope
    LoDTensor* gpu_tensor = pipeline_scope->Var(name)->GetMutable<LoDTensor>();
    TensorCopy(*static_cast<const Tensor*>(&root_tensor), place_, *dev_ctx_,
        static_cast<Tensor*>(gpu_tensor));
  }
}

void PipeSection::AutoSetCPUAffinity(bool reuse) {
	static int cpu_id = -1;
	static std::mutex mutex;

  mutex.lock();
  ++cpu_id;
  // thread-safe local parameter
  int thread_cpu_id = cpu_id;
  mutex.unlock();

  unsigned concurrency_cap = std::thread::hardware_concurrency();

  unsigned proc = thread_cpu_id;
  if (proc >= concurrency_cap) {
    if (reuse) {
      proc %= concurrency_cap;
    } else {
      LOG(INFO) << "All " << concurrency_cap
        << " CPUs have been set affinities. Fail to set " << thread_cpu_id << "th thread";
      return;
    }
  }

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(proc, &mask);

  if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
    return;
  }

  CPU_ZERO(&mask);
  if ((0 != sched_getaffinity(0, sizeof(mask), &mask)) || (0 == CPU_ISSET(proc, &mask))) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
  }

  LOG(INFO) << "Set " << thread_cpu_id << "th thread affinity to CPU " << proc;
}

void PipeSection::RetrieveSyncParamNames(std::vector<std::string>* param_names) {
  PADDLE_ENFORCE_NOT_NULL(param_names, "Invalid parameter");
  if (!platform::is_gpu_place(place_)) {
    return;
  }
  param_names->insert(param_names->end(), param_names_.begin(), param_names_.end());
}

// TODO: make processing steps dynamic
// TODO: print fetch vars if needed
void PipeSection::Start(std::vector<std::thread>* pipe_threads) {
#ifdef PIPELINE_PROFILE
  stats_.resize(concurrency_);
#endif

  concurrency_monitor_ = concurrency_;

  for (int i = 0; i < concurrency_; ++i) {
    pipe_threads->push_back(std::thread([this, i]() {
      AutoSetCPUAffinity(true);
      cudaSetDevice(kRankId);

      long step_cnt = 0;
      long accum_num = 0;
      int batch_size = 0;
      Scope* scope = nullptr;

      PIPELINE_PROFILE_DECLARE(reader);
      PIPELINE_PROFILE_DECLARE(calc);
      PIPELINE_PROFILE_DECLARE(trans);
      PIPELINE_PROFILE_DECLARE(main);
      PIPELINE_PROFILE_DECLARE(outer);
      PIPELINE_PROFILE_DECLARE(sync);

      bool started = false;
      while (joint_in_->Receive(&scope)) {
        if (!started) {
          PIPELINE_PROFILE_START(outer);
          started = true;
        }

        PIPELINE_PROFILE_RESUME(main);

        if (read_func_) {
          PIPELINE_PROFILE_RESUME(reader);
          batch_size = (*read_func_)(scope, i);
          PIPELINE_PROFILE_PAUSE(reader);
#ifdef DEBUG
        SEC_V3LOG << "htj in read_func, the batchsize is: " << batch_size;
#endif

          if (batch_size <= 0) {
            break;
          }

          SEC_V3LOG << "read batch size " << batch_size;
        } else {
          // TODO: Keep batch_size in scope? Or is there a better way to fetch batch_size?
          PADDLE_ENFORCE(joint_in_var_names_.size(), "PipeSection without a reader or joint-in variable is not supported by now");
          const LoDTensor& tensor = scope->FindVar(joint_in_var_names_[0])->Get<LoDTensor>();
          batch_size = tensor.lod().size() ? tensor.lod()[0].size() - 1 : tensor.dims()[0];

          SEC_V3LOG << "input batch size " << batch_size;
        }

        Scope* exe_scope = scope;
        if (sec_index_ > 0 && platform::is_gpu_place(place_)) {
          SEC_V3LOG << "CPU2GPU memory copy";
          PIPELINE_PROFILE_RESUME(trans);

          if (scope->kids().empty()) {
            exe_scope = &scope->NewScope();
          } else {
            exe_scope = scope->kids().front();
            PADDLE_ENFORCE(scope->kids().size() == 1, "scope->kids().size(): %zu", scope->kids().size());
          }

          for (const std::string& name : joint_in_var_names_) {
						SEC_V3LOG << " varname: " << name;
            const LoDTensor& src_tensor = scope->FindVar(name)->Get<LoDTensor>();
            if (platform::is_gpu_place(src_tensor.place())) {
              continue;
            }

            LoDTensor* gpu_tensor = exe_scope->Var(name)->GetMutable<LoDTensor>();
				gpu_tensor->set_lod(src_tensor.lod());

            TensorCopy(*static_cast<const Tensor*>(&src_tensor), place_, *dev_ctx_,
                static_cast<Tensor*>(gpu_tensor));
						SEC_V3LOG << "copy succeed";
          }

          PIPELINE_PROFILE_PAUSE(trans);
        }


        PIPELINE_PROFILE_RESUME(calc);
#ifdef DEBUG
        fprintf(stderr, "htj prepare to run op in r:%d, section:%d, thread:%d\n", kRankId, sec_index_, i);
#endif
        for (auto& op : ops_) {
#ifdef DEBUG
        	fprintf(stderr, "htj in line:%d\n", __LINE__);
#endif
					SEC_V3LOG << op->Type() << " op start in place=[" << (platform::is_gpu_place(place_)?"GPU":"CPU") << "]";
#ifdef DEBUG
					SEC_V3LOG << op->Type() << " op inputs:";
					for (const std::string& name : op->InputVars()) {
						SEC_V3LOG << op->Type() << " op input=[" << name << "]";
						Variable* var = exe_scope->FindVar(name);
						platform::Place place;
						if (var->IsType<LoDTensor>()) {
							place = exe_scope->FindVar(name)->Get<LoDTensor>().place();
						} else if (var->IsType<SelectedRows>()) {
							place = exe_scope->FindVar(name)->Get<SelectedRows>().place();
						} else {
							PADDLE_ENFORCE(false, "unexpected variable type");
						}
						SEC_V3LOG << op->Type() << " op input=[" << name << "] place=[" << (platform::is_gpu_place(place)?"GPU":"CPU") << "]";
						//SEC_V3LOG << op->Type() << " op input=[" << name << "] place=[" << (platform::is_gpu_place(place)?"GPU":"CPU") << "]";
					}
#endif
          op->Run(*exe_scope, place_);
        }
#ifdef DEBUG
        fprintf(stderr, "htj succeed to run op in r:%d, section:%d, thread:%d\n", kRankId, sec_index_, i);
#endif
        exe_scope->DropKids();
        // Wait for GPU calc finising, as the cudaMemcpy and GPU calc may be in different streams
        // No effect when it is a CPUDeviceContext
#ifdef DEBUG
        SEC_V3LOG << "htj before dev_ctx->wait";
#endif
        dev_ctx_->Wait();
#ifdef DEBUG
        SEC_V3LOG << "htj after dev_ctx->wait";
#endif
        PIPELINE_PROFILE_PAUSE(calc);

        if (next_ != nullptr && platform::is_gpu_place(place_)) {
          // FIXME: Temporarily we assume two adjacent sections are in different places,
          // and we do data transformation only in sections in GPU place, so the data is
          // transform from GPU to CPU
          // A better way to handle such a data transformation is to record each place of
          // joint-out variables, and do transform as required

          SEC_V3LOG << "GPU2CPU memory copy";
          PIPELINE_PROFILE_RESUME(trans);

          for (const std::string& name : joint_out_var_names_) {
            const LoDTensor& src_tensor = exe_scope->FindVar(name)->Get<LoDTensor>();
            LoDTensor* dst_tensor = scope->Var(name)->GetMutable<LoDTensor>();
			dst_tensor->set_lod(src_tensor.lod());

            TensorCopy(*static_cast<const Tensor*>(&src_tensor), next_->place_, *dev_ctx_,
                static_cast<Tensor*>(dst_tensor));
          }

          PIPELINE_PROFILE_PAUSE(trans);
        }

#ifdef DEBUG
        SEC_V3LOG << "htj send scope to next section successfully";
#endif
        joint_out_->Send(scope);

        if (sync_func_) {
          PIPELINE_PROFILE_RESUME(sync);
          (*sync_func_)(scope, i);
          PIPELINE_PROFILE_PAUSE(sync);
        }

        ++step_cnt;
        accum_num += batch_size;

        PIPELINE_PROFILE_PAUSE(main);
      }
      PIPELINE_PROFILE_PAUSE(outer);

      concurrency_mutex_.lock();
      --concurrency_monitor_;
#ifdef DEBUG
        fprintf(stderr, "htj in line%d, close joint out\n", __LINE__);
        SEC_V3LOG << "htj minus --concurrency_monitor_ to " << concurrency_monitor_;
#endif
      concurrency_mutex_.unlock();

      if (concurrency_monitor_ <= 0) {
#ifdef DEBUG
          SEC_V3LOG << "htj in exit function ";
#endif
        while (sec_index_ < sec_num_-1 && joint_out_->Size()) {
          sleep(1);
        }
#ifdef DEBUG
          SEC_V3LOG << "htj close";
#endif
        joint_out_->Close();
      }

#ifdef PIPELINE_PROFILE
      // change to map container
      stats_[i].reader_ratio = reader_timer.ElapsedUS() / outer_timer.ElapsedUS();
      stats_[i].reader_us = reader_timer.ElapsedUS() / step_cnt;
      stats_[i].reader_throughput =
        reader_timer.Count() ? accum_num / reader_timer.ElapsedSec() : 0;
      stats_[i].trans_ratio = trans_timer.ElapsedUS() / outer_timer.ElapsedUS();
      stats_[i].trans_us = trans_timer.ElapsedUS() / step_cnt;
      stats_[i].trans_throughput = accum_num / trans_timer.ElapsedSec();
      stats_[i].calc_ratio = calc_timer.ElapsedUS() / outer_timer.ElapsedUS();
      stats_[i].calc_us = calc_timer.ElapsedUS() / step_cnt;
      stats_[i].calc_throughput = accum_num / calc_timer.ElapsedSec();
      stats_[i].sync_ratio = sync_timer.ElapsedUS() / outer_timer.ElapsedUS();
      stats_[i].sync_us = sync_timer.ElapsedUS() / step_cnt;
      stats_[i].sync_throughput =
        sync_timer.Count() ? accum_num / sync_timer.ElapsedSec() : 0;
      stats_[i].main_ratio = main_timer.ElapsedUS() / outer_timer.ElapsedUS();
      stats_[i].main_us = main_timer.ElapsedUS() / step_cnt;
      stats_[i].main_throughput = accum_num / main_timer.ElapsedSec();
      stats_[i].outer_throughput = accum_num / outer_timer.ElapsedSec();
      stats_[i].instance_num = accum_num;
#endif
    }));
  }
}

#define PL_V3LOG LOG(ERROR) << "[r" << rank_id_ << "] "
//#define PL_V3LOG VLOG(3) << "[r" << rank_id_ << "] "

void PipelineWorker::Initialize(const TrainerDesc& desc) {
  const PipelineWorkerParameter& sec_param = desc.pipeline_param();
  nscopes_ = sec_param.context_scope_num();
  PL_V3LOG << "context scope number: " << nscopes_;

  int sec_num = sec_param.pipe_sec_cfg_size();
  PADDLE_ENFORCE(sec_num > 1, "Too few pipe sections");
  PL_V3LOG << "actual section number: " << sec_num;

  std::vector<std::string> param_names;
  PADDLE_ENFORCE(sections_.empty(), "pipe section vector is not empty before initialization");
  for (int i = 0; i < sec_num; ++i) {
    sections_.emplace_back(new PipeSection(sec_param.pipe_sec_cfg(i), rank_id_));
    sections_[i]->RetrieveSyncParamNames(&param_names);

    sections_[i]->SetSectionIndex(i);
    sections_[i]->SetSectionNum(sec_num);
    sections_[i]->CreateOperators();

    if (i == 0) {
      // readers are assigned to the first section
      read_func_.reset(new ReadFunctor(device_readers_));
      sections_[i]->SetReadFunctor(read_func_.get());
    }
  }

  for (int i = 0; i < sec_num; ++i) {
    sections_[i]->SetPrePipeSection(i == 0 ? nullptr : sections_[i - 1].get());
    sections_[i]->SetNextPipeSection(i == int(sections_.size() - 1) ? nullptr : sections_[i + 1].get());
  }

  nasync_steps_ = sec_param.nasync_steps();
  PL_V3LOG << "asynchronous training steps: " << nscopes_;
  /*
  if (nranks_ > 1) {
    SynchronizeFunctor* func = new SynchronizeFunctor(rank_id_, nranks_, nasync_steps_);
    func->SetSyncParamNames(param_names);
    sync_func_.reset(func);
    for (int i = sec_num - 1; i >= 0; --i) {
      if (platform::is_gpu_place(sections_[i]->place())) {
        sections_[i]->SetSynchronizeFunctor(sync_func_.get());
        break;
      }
    }
  }
  */
}

void PipelineWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  main_program_.reset(new ProgramDesc(main_prog));

  PADDLE_ENFORCE(root_scope_, "Null root_scope pointer");
  pipeline_scope_ = &root_scope_->NewScope();

  PADDLE_ENFORCE(joints_.empty(), "Joints vector is not empty before initialization");
  joints_.emplace_back(new PipeJoint(nscopes_));
  PipeJoint* scope_pool = joints_.front().get();
  InitScopePool(scope_pool);

  std::vector<std::string> param_names;
  for (size_t i = 0; i < sections_.size(); ++i) {
    PipeJoint* joint_in = joints_.back().get();
    PipeJoint* joint_out = nullptr;
    if (i < sections_.size() - 1) {
      joints_.emplace_back(new PipeJoint(nscopes_));
      joint_out = joints_.back().get();
    } else {
      joint_out = joints_.front().get();
    }
    sections_[i]->SetJoint(joint_in, joint_out);
    sections_[i]->RetrieveSyncParamNames(&param_names);
    sections_[i]->CopyParameters(*root_scope_, pipeline_scope_);
  }
  if (nranks_ > 1) {
    SynchronizeFunctor* func = new SynchronizeFunctor(rank_id_, nranks_, nasync_steps_);
    func->SetSyncParamNames(param_names);
    //func->pipeline_scopes_[rank_id_] = pipeline_scope_;
    func->SetPipelineScope(rank_id_, pipeline_scope_);
    sync_func_.reset(func);
    for (int i = sections_.size() - 1; i >= 0; --i) {
      if (platform::is_gpu_place(sections_[i]->place())) {
        sections_[i]->SetSynchronizeFunctor(sync_func_.get());
        break;
      }
    }
  }
}

void PipelineWorker::InitScopePool(PipeJoint* scope_pool) {
  cudaSetDevice(rank_id_);
  for (int i = 0; i < nscopes_; ++i) {
    Scope* scope = &pipeline_scope_->NewScope();
    for (auto& var : main_program_->Block(0).AllVars()) {
#ifdef DEBUG
        //fprintf(stderr, "htj in %s: pipe scope: varname: %s\n", __FUNCTION__, var->Name().c_str());
#endif
      if (!var->Persistable()) {
#ifdef DEBUG
        //fprintf(stderr, "htj in %s: non-persistable varname: %s\n", __FUNCTION__, var->Name().c_str());
#endif
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }

    scope_pool->Send(scope);
  }
}

void PipelineWorker::TrainFiles() {
  for (auto& sec : sections_) {
    sec->Start(&pipe_threads_);
  }

  for (auto& t : pipe_threads_) {
    t.join();
  }
  PL_V3LOG << "train succeed";

  // log performance
#ifdef PIPELINE_PROFILE
  for (int i = 0; i < sections_.size(); ++i) {
    const auto &stat = sections_[i]->GetStats();

    PipeSection::ProfileStat summary_stat;
    size_t con_num = stat.size();
    for (int j = 0; j < con_num; ++j) {
      summary_stat.reader_ratio += stat[j].reader_ratio  / con_num;
      summary_stat.reader_us += stat[j].reader_us  / con_num;
      summary_stat.reader_throughput += stat[j].reader_throughput;
      summary_stat.trans_ratio += stat[j].trans_ratio  / con_num;
      summary_stat.trans_us += stat[j].trans_us  / con_num;
      summary_stat.trans_throughput += stat[j].trans_throughput;
      summary_stat.calc_ratio += stat[j].calc_ratio  / con_num;
      summary_stat.calc_us += stat[j].calc_us  / con_num;
      summary_stat.calc_throughput += stat[j].calc_throughput;
      summary_stat.sync_ratio += stat[j].sync_ratio  / con_num;
      summary_stat.sync_us += stat[j].sync_us  / con_num;
      summary_stat.sync_throughput += stat[j].sync_throughput;
      summary_stat.main_ratio += stat[j].main_ratio  / con_num;
      summary_stat.main_us += stat[j].main_us  / con_num;
      summary_stat.main_throughput += stat[j].main_throughput;
      summary_stat.outer_throughput += stat[j].outer_throughput;
      summary_stat.instance_num += stat[j].instance_num;
    }
    PL_V3LOG << "profile for section: " << i;
    PL_V3LOG << "reader throughput: " << summary_stat.reader_throughput;
    PL_V3LOG << "main throughput: " << summary_stat.main_throughput;
  }
#endif
}

}  // namespace framework
}  // namespace paddle
