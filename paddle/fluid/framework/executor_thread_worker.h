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
#include "nccl.h"

namespace paddle {
namespace framework {

void CreateTensor(Variable* var, proto::VarType::Type var_type);

class ExecutorThreadWorker {
 public:
  ExecutorThreadWorker(int nranks, int rank_id, int nscopes, int ncpu_calc_threads,
      int nasync_steps, Scope* scope, const ProgramDesc& main_program_desc,
      const std::vector<std::shared_ptr<DataFeed>>& readers,
      const std::vector<std::string>& fetch_var_names) 
        : nranks_(nranks), rank_id_(rank_id), nscopes_(nscopes),
        ncpu_calc_threads_(ncpu_calc_threads), nasync_steps_(nasync_steps),
        readers_(readers), root_scope_(scope) {
    main_program_.reset(new ProgramDesc(main_program_desc));
    fetch_var_names_.insert(fetch_var_names_.end(), fetch_var_names.begin(),
                            fetch_var_names.end());
    cpu_place_ = platform::default_cpu();
    gpu_place_ = platform::CUDAPlace(rank_id);
    cpu_dev_ctx_.reset(new platform::CPUDeviceContext(cpu_place_));
    gpu_dev_ctx_.reset(new platform::CUDADeviceContext(gpu_place_));
  }

  virtual ~ExecutorThreadWorker() {}

  void CreateThreadResource(const framework::ProgramDesc& program);

  // A multi-thread training function
  virtual void TrainFiles();

 private:
  void CreateThreadScope(const framework::ProgramDesc& program);
  void CreateThreadOperators(const framework::ProgramDesc& program);

  void LookupTable(Scope* scope);
  void LookupTableGrad(Scope* scope);
  void LookupTableSumConcat(Scope* scope);
  void LookupTableSumConcatGrad(Scope* scope);
  void StartReaders();
  void StartEmbFFThreads();
  void StartGPUCalcThread();
  void StartEmbBPThreads();
  void AsyncUpdateParam();

 protected:
  int nranks_;
  int rank_id_;
  int nscopes_;
  int ncpu_calc_threads_;
  int nasync_steps_;

  cudaStream_t cuda_stream_;

  std::vector<std::shared_ptr<DataFeed>> readers_;
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

  int64_t padding_idx_{-1};
};

}  // namespace framework
}  // namespace paddle
