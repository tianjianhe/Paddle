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

#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace framework {

class PullDenseWorker {
 public:
  virtual ~PullDenseWorker() {}
  virtual void Initialize(const TrainerDesc& param);
  int Start();
  void Stop();
  void SetRootScope(Scope* scope) { root_scope_ = scope; }
  void IncreaseThreadVersion(int thread_id, uint64_t table_id);
  void ResetThreadVersion(uint64_t table_id);
  void Wait(std::vector<::std::future<int32_t>>* status_vec);
  static std::shared_ptr<PullDenseWorker> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::PullDenseWorker());
    }
    return s_instance_;
  }

 private:
  PullDenseWorker() : root_scope_(NULL) {}
  void Run();
  bool CheckUpdateParam(uint64_t table_id);

 private:
  static std::shared_ptr<PullDenseWorker> s_instance_;
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  PullDenseWorkerParameter param_;
  DownpourWorkerParameter dwp_param_;
  Scope* root_scope_;
  bool running_;

  static std::map<uint64_t, uint64_t> last_versions_;
  static std::map<uint64_t, uint64_t> current_version_;
  static std::mutex mutex_for_version_;
  static std::map<uint64_t, std::vector<uint64_t>> training_versions_;
  static std::map<uint64_t, std::vector<std::string>> dense_value_names_;

  std::thread t_;
  int thread_num_;
  int sleep_time_ms_;
  int threshold_;

  std::vector<::std::future<int32_t>> pull_dense_status_;
  uint32_t pull_dense_fail_times_ = 0;
  std::vector<float> base_norm_param_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  float squared_sum_epsilon_ = 1e-4;
  std::mutex mutex_for_mean_scale_;
  float total_batch_num_ = 0;
};

// should incorporate different type of device
class DeviceWorker {
 public:
  DeviceWorker() {}
  virtual ~DeviceWorker() {}
  // OPTIMIZE: it is better to initialize with configs like DeviceWorkerDesc
  // to decouple with Trainer
  virtual void Initialize(const TrainerDesc& desc) = 0;
  virtual void SetDeviceIndex(int tid) = 0;
  virtual void SetDeviceNum(int device_num) = 0;
  virtual void TrainFiles() = 0;
  virtual void PrintFetchVars() = 0;
  virtual void TrainFilesWithProfiler() = 0;
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) = 0;
  // will make this zero copy in the future
  virtual void BindingDataFeedMemory() = 0;
  virtual void SetRootScope(Scope* root_scope);
  virtual void SetDataFeed(const std::shared_ptr<DataFeed>& data_feed);
  // The device worker may reserve multiple readers
  virtual void SetDataFeed(std::vector<std::shared_ptr<DataFeed>>::iterator begin, size_t num);
  virtual void SetPlace(const paddle::platform::Place& place) {
    place_ = place;
  }

 protected:
  Scope* root_scope_;
  paddle::platform::Place place_;
  std::shared_ptr<DataFeed> device_reader_;
  std::vector<std::shared_ptr<DataFeed>> device_readers_;
  int64_t batch_num_;
  FetchConfig fetch_config_;
};

class CPUWorkerBase : public DeviceWorker {
 public:
  CPUWorkerBase() {}
  virtual ~CPUWorkerBase() {}
  virtual void SetDeviceIndex(int tid) { thread_id_ = tid; }
  virtual void SetDeviceNum(int device_num) { thread_num_ = device_num; };
  virtual void TrainFiles() = 0;
  virtual void TrainFilesWithProfiler() {}
  virtual void PrintFetchVars() {}
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) {}

 protected:
  int thread_id_;
  int thread_num_;
};

class HogwildWorker : public CPUWorkerBase {
 public:
  HogwildWorker() {}
  virtual ~HogwildWorker() {}
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void PrintFetchVars();
  virtual void CreateDeviceResource(const ProgramDesc& main_prog);
  virtual void BindingDataFeedMemory();

 protected:
  void CreateThreadOperators(const ProgramDesc& program);
  void CreateThreadScope(const ProgramDesc& program);
  std::vector<std::string> op_names_;
  std::vector<OperatorBase*> ops_;
  Scope* thread_scope_;
  HogwildWorkerParameter param_;
  std::vector<std::string> skip_ops_;
};

class DownpourWorker : public HogwildWorker {
 public:
  DownpourWorker() {}
  virtual ~DownpourWorker() {}
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();

 protected:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  std::shared_ptr<paddle::framework::PullDenseWorker> pull_dense_worker_;
  void FillSparseValue(size_t table_id);
  void PushGradients();
  void CollectLabelInfo(size_t table_id);

 private:
  bool need_to_push_dense_;
  bool need_to_push_sparse_;
  DownpourWorkerParameter param_;
  // just save the value in param_ for easy access
  std::map<uint64_t, std::string> label_var_name_;
  std::map<uint64_t, std::vector<std::string>> sparse_key_names_;
  std::map<uint64_t, std::vector<std::string>> sparse_value_names_;
  std::map<uint64_t, std::vector<std::string>> sparse_grad_names_;
  std::map<uint64_t, std::vector<std::string>> dense_value_names_;
  std::map<uint64_t, std::vector<std::string>> dense_grad_names_;

  // feasign
  std::map<uint64_t, std::vector<uint64_t>> features_;
  // feasign stats
  std::map<uint64_t, std::vector<float>> feature_labels_;
  // feasign embedding
  std::map<uint64_t, std::vector<std::vector<float>>> feature_values_;
  // feasign embedding gradient
  std::map<uint64_t, std::vector<std::vector<float>>> feature_grads_;
  // skipped ops
  std::vector<std::string> skip_ops_;

  std::shared_ptr<PullDenseWorker> _pull_dense_worker;
  std::vector<::std::future<int32_t>> push_sparse_status_;
  std::vector<::std::future<int32_t>> push_dense_status_;
};

using PipeJoint = operators::reader::BlockingQueue<Scope*>;

class PipelineFunctor {
 public:
  PipelineFunctor() {}
  virtual ~PipelineFunctor() {}

  virtual int operator()(Scope* scope, int index) = 0;
};

class ReadFunctor : public PipelineFunctor {
 public:
  ReadFunctor(const std::vector<std::shared_ptr<DataFeed>>& readers) {
    readers_ = readers;
  }
  virtual ~ReadFunctor() {}

  int operator()(Scope* scope, int index) override;

 protected:
  std::vector<std::shared_ptr<DataFeed>> readers_;

};

class SynchronizeFunctor : public PipelineFunctor {
 public:
  SynchronizeFunctor(int rank_id, int rank_num, int nasync_steps);
  virtual ~SynchronizeFunctor() {}

  void SetSyncParamNames(const std::vector<std::string>& param_names) {
    sync_param_names_ = param_names;
  }

  int operator()(Scope* scope, int index) override;
  //static std::vector<Scope*>	pipeline_scopes_;
  void SetPipelineScope(int k, Scope *pipeline_scope) {
    PADDLE_ENFORCE(k < pipeline_scopes_.size(), "exceed number of card");
    pipeline_scopes_[k] = pipeline_scope;
  }
  static void remove_nccl_map() {
    nccl_ctx_map_.reset(nullptr);
  }

 protected:
  static std::unique_ptr<platform::NCCLContextMap> nccl_ctx_map_;
  //static std::unique_ptr<ncclUniqueId> nccl_id_;

  uint64_t sync_signal_;

  const int kRankId;
  const int kNRanks;

  const int kNAsyncSteps;

  static uint64_t s_sync_flag_;

  int counter_;

  std::vector<std::string> sync_param_names_;
  static std::vector<Scope*>	pipeline_scopes_;

  void Synchronize();

};

class PipeSection {
 public:
  explicit PipeSection(const PipeSectionConfig& cfg, int rank_id);
  ~PipeSection() {}

  const platform::Place& place() const { return place_; }

  int sec_index() const { return sec_index_; }
  void SetSectionIndex(int index) { sec_index_ = index;	}

  int sec_num() const { return sec_num_; }
  void SetSectionNum(int sec_num) { sec_num_ = sec_num;	}

  const PipeSection* pre() const { return pre_; }
  void SetPrePipeSection(PipeSection* sec) { pre_ = sec; }

  const PipeSection* next() const { return next_; }
  void SetNextPipeSection(PipeSection* sec) { next_ = sec; }

  void SetJoint(PipeJoint* joint_in, PipeJoint* joint_out) {
    joint_in_ = joint_in;
    joint_out_ = joint_out;
  }

  void SetReadFunctor(PipelineFunctor* read_func) {
    read_func_ = read_func;
  }

  void SetSynchronizeFunctor(PipelineFunctor* sync_func) {
    sync_func_ = sync_func;
  }

  void Start(std::vector<std::thread>* pipe_threads);

  void CopyParameters(const Scope& root_scope, Scope* pipeline_scope);

  void CreateOperators();

  void RetrieveSyncParamNames(std::vector<std::string>* param_vars);

  struct ProfileStat {
    double reader_ratio = 0;
    double reader_us = 0;
    double reader_throughput = 0;
    double trans_ratio = 0;
    double trans_us = 0;
    double trans_throughput = 0;
    double calc_ratio = 0;
    double calc_us = 0;
    double calc_throughput = 0;
    double sync_ratio = 0;
    double sync_us = 0;
    double sync_throughput = 0;
    double main_ratio = 0;
    double main_us = 0;
    double main_throughput = 0;
    double outer_throughput = 0;
    double instance_num = 0;
  };

  std::vector<ProfileStat> GetStats() const {return stats_;}
 protected:
  const int kRankId;

	int sec_index_;
  int sec_num_;

  PipeSection* pre_;
  PipeSection* next_;

  PipelineFunctor* read_func_;
  PipelineFunctor* sync_func_;

  std::shared_ptr<framework::ProgramDesc> program_;

	platform::Place place_;
	platform::DeviceContext* dev_ctx_;

  int concurrency_;

  PipeJoint* joint_in_;
  PipeJoint* joint_out_;

  std::vector<std::string> joint_in_var_names_;
  std::vector<std::string> joint_out_var_names_;
  std::vector<std::string> param_names_;

  int concurrency_monitor_;
  std::mutex concurrency_mutex_;

  std::vector<std::unique_ptr<OperatorBase>> ops_;

  std::vector<ProfileStat> stats_;

  void Postprocess(Scope* scope);

  void AutoSetCPUAffinity(bool reuse = true);

};

class PipelineWorker : public DeviceWorker {
 public:
  PipelineWorker() {}
  virtual ~PipelineWorker() {}

  virtual void SetDeviceIndex(int tid) { rank_id_ = tid; }
  virtual void SetDeviceNum(int device_num) { nranks_ = device_num; }

  virtual void Initialize(const TrainerDesc& desc) override;

  virtual void CreateDeviceResource(const ProgramDesc& main_prog) override;

  virtual void BindingDataFeedMemory() override {}

  virtual void TrainFiles() override;
  virtual void TrainFilesWithProfiler() override {}

  virtual void PrintFetchVars() override {};

 protected:
  int nscopes_;

  int nranks_;
  int rank_id_;

  int nasync_steps_;

  std::unique_ptr<PipelineFunctor> read_func_;
  std::unique_ptr<PipelineFunctor> sync_func_;

  std::unique_ptr<framework::ProgramDesc> main_program_;

  Scope* pipeline_scope_;

	std::vector<std::unique_ptr<PipeSection>> sections_;

  std::vector<std::unique_ptr<PipeJoint>> joints_;

  std::vector<std::thread> pipe_threads_;

  virtual void InitScopePool(PipeJoint* scope_pool);

  virtual void InitPipelineInner() {};

};

}  // namespace framework
}  // namespace paddle
