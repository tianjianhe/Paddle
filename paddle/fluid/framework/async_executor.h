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

#include <time.h>
#include <map>
#include <memory>
#include <mutex>   // NOLINT
#include <random>  // local_random_engine
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <typeinfo>
#include <vector>
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "nccl.h"

namespace paddle {
namespace framework {

class DataFeed;
class ExecutorThreadWorker;

class AsyncExecutor {
 public:
  AsyncExecutor(Scope* scope, const platform::Place& place)
      : root_scope_(scope), place_(place), sync_signal_(0) {}
  virtual ~AsyncExecutor() {}

  void RunFromFile(const ProgramDesc& main_program,
                   const std::string& data_feed_desc_str,
                   const std::vector<std::string>& filelist,
                   const std::vector<std::string>& fetch_names,
                   const int ncards,
                   const int nscopes,
                   const int nreaders,
                   const int nemb_ff_threads,
                   const int nemb_bp_threads,
                   const int nasync_steps);

  void UpdateSyncFlag(int rank_id);
  
 private:
  void InitRootScope(const ProgramDesc& program);
  void PrepareReaders(std::vector<std::vector<std::shared_ptr<DataFeed>>>& readers,
                      int ncards,
                      int nreaders,
                      const DataFeedDesc& data_feed_desc,
                      const std::vector<std::string>& filelist);

 public:
  Scope* root_scope_;
  platform::Place place_;

 private:
  std::mutex sync_signal_mutex_;
  uint64_t sync_flag_;
  uint64_t reset_sync_flag_;
  const uint64_t sync_signal_;
  std::vector<std::shared_ptr<ExecutorThreadWorker>> workers_;

};

}  // namespace framework
}  // namespace paddle
