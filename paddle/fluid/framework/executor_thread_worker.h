/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/operators/math/blas.h"
#include "nccl.h"

namespace paddle {
namespace framework {

class AsyncExecutor;

void CreateTensor(Variable* var, proto::VarType::Type var_type);

class ExecutorThreadWorker {
 public:
  ExecutorThreadWorker(int nranks, int rank_id, int nscopes,
      int nemb_ff_threads, int nemb_bp_threads,
      int nasync_steps, Scope* scope, const ProgramDesc& main_program_desc,
      const std::vector<std::shared_ptr<DataFeed>>& readers,
      const std::vector<std::string>& fetch_var_names,
      ncclComm_t* nccl_comm, AsyncExecutor* exe) 
        : nranks_(nranks), rank_id_(rank_id), nscopes_(nscopes),
        nemb_ff_threads_(nemb_ff_threads), nemb_bp_threads_(nemb_bp_threads),
        nasync_steps_(nasync_steps), readers_(readers), root_scope_(scope),
        nccl_comm_(nccl_comm), exe_(exe) {
    main_program_.reset(new ProgramDesc(main_program_desc));
    fetch_var_names_.insert(fetch_var_names_.end(), fetch_var_names.begin(),
                            fetch_var_names.end());
    cpu_place_ = platform::default_cpu();
    gpu_place_ = platform::CUDAPlace(rank_id);
    cpu_dev_ctx_.reset(new platform::CPUDeviceContext(cpu_place_));
    gpu_dev_ctx_.reset(new platform::CUDADeviceContext(gpu_place_));
    cpu_blas_.reset(new operators::math::BlasT<platform::CPUDeviceContext, float>(*cpu_dev_ctx_));
    gpu_blas_.reset(new operators::math::BlasT<platform::CUDADeviceContext, float>(*gpu_dev_ctx_));
    
    sync_signal_ = false;

    cudaSetDevice(rank_id_);
    cudaStreamCreate(&cuda_stream_);
  }

  virtual ~ExecutorThreadWorker() {
    cudaStreamDestroy(cuda_stream_);
  }

  void CreateThreadResource();

  // A multi-thread training function
  virtual void TrainFiles();

  void SetSyncSignal() { sync_signal_ = true; }

 private:
  void CreateThreadScope();
  void CreateThreadOperators();

  void LookupTable(Scope* scope);
  void LookupTableGrad(Scope* scope);
  void LookupTableSumConcat(Scope* scope);
  void LookupTableSumConcatGrad(Scope* scope);
  void StartReaders();
  void StartEmbFFThreads();
  void StartEmbBPThreads();
  void StartGPUCalc();
  //void AsyncUpdateParam();
  void SyncParam();
  void LogFetchValues(const Scope& scope);

 protected:
  int nranks_;
  int rank_id_;
  int nscopes_;
  int nemb_ff_threads_;
  int nemb_bp_threads_;
  int nasync_steps_;
  bool sync_signal_;

  cudaStream_t cuda_stream_;

  std::vector<std::shared_ptr<DataFeed>> readers_;
  int reader_num_monitor_;
  std::mutex reader_num_mutex_;

  // operator name
  std::vector<std::string> op_names_;
  // thread level, local operators for forward and backward
  std::vector<OperatorBase*> ops_;
  // main program for training
  std::unique_ptr<framework::ProgramDesc> main_program_;
  // execution place
  std::shared_ptr<platform::CUDADeviceContext> gpu_dev_ctx_;
  std::shared_ptr<platform::CPUDeviceContext> cpu_dev_ctx_;
  platform::CPUPlace cpu_place_;
  platform::CUDAPlace gpu_place_;
  std::unique_ptr<operators::math::BlasT<platform::CPUDeviceContext, float>> cpu_blas_;
  std::unique_ptr<operators::math::BlasT<platform::CUDADeviceContext, float>> gpu_blas_;

  // root scope for model parameters
  Scope* root_scope_;
  std::shared_ptr<Scope> thread_scope_;
  std::shared_ptr<Scope> param_scope_;
  std::shared_ptr<operators::reader::BlockingQueue<DataFeed*>> reader_queue_;
  std::shared_ptr<operators::reader::BlockingQueue<Scope*>> emb_ff_scope_queue_;
  std::shared_ptr<operators::reader::BlockingQueue<Scope*>> emb_bp_scope_queue_;
  std::shared_ptr<operators::reader::BlockingQueue<Scope*>> scope_pool_;

  std::vector<std::string> fetch_var_names_;
  std::vector<std::vector<float>> fetch_values_;
  std::vector<std::thread> all_threads_;
  std::vector<std::string> ids_names_;
  std::vector<std::string> param_names_;

  ncclComm_t* nccl_comm_;
  AsyncExecutor* exe_;

  uint64_t padding_idx_{0};

  struct EmbFFStat {
    double reader_ratio = 0;
    double reader_us = 0;
    double reader_throughput = 0;
    double emb_ff_ratio = 0;
    double emb_ff_us = 0;
    double throughput = 0;
  };
  struct MainNetStat {
    double memcpy_ratio = 0;
    double memcpy_us = 0;
    double memcpy_trp = 0;
    double gpu_ratio = 0;
    double gpu_us = 0;
    double gpu_trp = 0;
    double other_ratio = 0;
    double other_us = 0;
    double sync_ratio = 0;
    double sync_us = 0;
    double main_net_ratio = 0;
    double main_net_us = 0;
    double main_net_trp = 0;
    double throughput = 0;
  };
  struct EmbBPStat {
    double emb_bp_ratio = 0;
    double emb_bp_us = 0;
    double throughput = 0;
  };

  std::vector<EmbFFStat> emb_ff_stats_;
  MainNetStat main_net_stat_;
  std::vector<EmbBPStat> emb_bp_stats_;

};

}  // namespace framework
}  // namespace paddle
