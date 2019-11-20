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

#pragma once

#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <ctime>
#include <vector>
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/platform/timer.h"
#ifdef PADDLE_WITH_BOX_PS
#include <boxps_public.h>
#endif
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

#define PADDLEBOX_LOG VLOG(0) << "PaddleBox: "
class BasicAucCalculator {
 public:
  BasicAucCalculator() {}
  void init(int table_size) {
    set_table_size(table_size);
    pass_id = 1;
  }
  void reset() {
    for (int i = 0; i < 2; i++) {
      _table[i].assign(_table_size, 0.0);
    }
    _local_abserr = 0;
    _local_sqrerr = 0;
    _local_pred = 0;
  }
  void add_data(double pred, int label) {
    PADDLE_ENFORCE(pred >= 0.0, "pred should be greater than 0");
    PADDLE_ENFORCE(pred <= 1.0, "pred should be lower than 1");
    PADDLE_ENFORCE((label == 0 || label == 1),
                   "label must be equal to 0 or 1, but its value is: %d",
                   label);
    int pos = std::min(static_cast<int>(pred * _table_size), _table_size - 1);
    PADDLE_ENFORCE(pos >= 0 && pos < _table_size,
                   "pos must in range [0, _table_size)");
    std::lock_guard<std::mutex> lock(_table_mutex);
    _local_abserr += fabs(pred - label);
    _local_sqrerr += (pred - label) * (pred - label);
    _local_pred += pred;
    _table[label][pos]++;
  }
  void compute();
  int table_size() const { return _table_size; }
  double bucket_error() const { return _bucket_error; }
  double auc() const { return _auc; }
  double mae() const { return _mae; }
  double actual_ctr() const { return _actual_ctr; }
  double predicted_ctr() const { return _predicted_ctr; }
  double size() const { return _size; }
  double rmse() const { return _rmse; }
  std::vector<double>& get_negative() { return _table[0]; }
  std::vector<double>& get_postive() { return _table[1]; }
  double& local_abserr() { return _local_abserr; }
  double& local_sqrerr() { return _local_sqrerr; }
  double& local_pred() { return _local_pred; }
  void calculate_bucket_error();
  int pass_id;

 protected:
  double _local_abserr = 0;
  double _local_sqrerr = 0;
  double _local_pred = 0;
  double _auc = 0;
  double _mae = 0;
  double _rmse = 0;
  double _actual_ctr = 0;
  double _predicted_ctr = 0;
  double _size;
  double _bucket_error = 0;

 private:
  void set_table_size(int table_size) {
    _table_size = table_size;
    for (int i = 0; i < 2; i++) {
      _table[i] = std::vector<double>();
    }
    reset();
  }
  int _table_size;
  std::vector<double> _table[2];
  static constexpr double kRelativeErrorBound = 0.05;
  static constexpr double kMaxSpan = 0.01;
  std::mutex _table_mutex;
};

class BoxWrapper {
 public:
  virtual ~BoxWrapper() {}
  BoxWrapper() {}

  void FeedPass(int date, const std::vector<uint64_t>& feasgin_to_box) const;
  void BeginPass() const;
  void EndPass() const;
  void ResetClickNum();
  void UpdateClickNum(int64_t actual_val, float pred_val);
  void PrintClickNum() const;
  void PullSparse(const paddle::platform::Place& place,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size);
  void InitializeGPU(const char* conf_file) {
    if (nullptr != s_instance_) {
      PADDLEBOX_LOG << "Begin InitializeGPU";
#ifdef PADDLE_WITH_BOX_PS
      std::vector<cudaStream_t*> stream_list;
      for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
        VLOG(3) << "before get context i[" << i << "]";
        platform::CUDADeviceContext* context =
            dynamic_cast<platform::CUDADeviceContext*>(
                platform::DeviceContextPool::Instance().Get(
                    platform::CUDAPlace(i)));
        PADDLEBOX_LOG << "after get cuda context for card [" << i << "]";
        PADDLE_ENFORCE_EQ(context == nullptr, false, "context is nullptr");

        stream_list_[i] = context->stream();
        stream_list.push_back(&stream_list_[i]);
      }
      PADDLEBOX_LOG << "call InitializeGPU in boxps";
      // the second parameter is useless
      s_instance_->boxps_ptr_->InitializeGPU(conf_file, -1, stream_list);
      PADDLEBOX_LOG << "return from InitializeGPU in boxps";
#endif
    }
  }
  void Finalize() {
    if (nullptr != s_instance_) {
      s_instance_->boxps_ptr_->Finalize();
    }
  }
  void SaveModel() const { printf("savemodel in box_Wrapper\n"); }
  void LoadModel() const { printf("loadmodel in box\n"); }

  static std::shared_ptr<BoxWrapper> GetInstance() {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, this lock can be removed
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        VLOG(3) << "s_instance_ is null";
        s_instance_.reset(new paddle::framework::BoxWrapper());
        s_instance_->cal_.reset(new BasicAucCalculator());
        s_instance_->ubm_cal_.reset(new BasicAucCalculator());
        s_instance_->day_join_cal_.reset(new BasicAucCalculator());
        s_instance_->day_update_cal_.reset(new BasicAucCalculator());
        s_instance_->day_ubm_join_cal_.reset(new BasicAucCalculator());
        s_instance_->day_ubm_update_cal_.reset(new BasicAucCalculator());
        s_instance_->cal_->init(1000000);
        s_instance_->ubm_cal_->init(1000000);
        s_instance_->day_join_cal_->init(1000000);
        s_instance_->day_update_cal_->init(1000000);
        s_instance_->day_ubm_join_cal_->init(1000000);
        s_instance_->day_ubm_update_cal_->init(1000000);
#ifdef PADDLE_WITH_BOX_PS
        s_instance_->boxps_ptr_.reset(boxps::BoxPSBase::GetIns());
#endif
      }
    }
    return s_instance_;
  }

 private:
#ifdef PADDLE_WITH_BOX_PS
  static cudaStream_t stream_list_[8];
  static std::shared_ptr<boxps::BoxPSBase> boxps_ptr_;
#endif
  static std::shared_ptr<BoxWrapper> s_instance_;
  int GetDate() const;
  // will be failed when multi datasets run concurrently.
  std::atomic<int64_t> actual_click;
  float pred_click;
  std::mutex add_mutex;

 public:
  int batch_size_;
  static std::shared_ptr<BasicAucCalculator> cal_;
  static std::shared_ptr<BasicAucCalculator> ubm_cal_;
  static std::shared_ptr<BasicAucCalculator> day_join_cal_;
  static std::shared_ptr<BasicAucCalculator> day_update_cal_;
  static std::shared_ptr<BasicAucCalculator> day_ubm_join_cal_;
  static std::shared_ptr<BasicAucCalculator> day_ubm_update_cal_;
};

class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset, int year, int month, int day) : dataset_(dataset), year_(year), month_(month), day_(day) {}
  virtual ~BoxHelper() {}

  void BeginPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
  }

  void EndPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass();
  }
  void LoadIntoMemory() {
    platform::Timer timer;
    PADDLEBOX_LOG << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
    timer.Start();
    dataset_->LoadIntoMemory();
    timer.Pause();
    PADDLEBOX_LOG << "download + parse cost: " << timer.ElapsedSec() << "s";

    timer.Start();
    FeedPass();
    timer.Pause();
    PADDLEBOX_LOG << "FeedPass cost: " << timer.ElapsedSec() << " s";
    PADDLEBOX_LOG << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
  }
  void PreLoadIntoMemory() {
    dataset_->PreLoadIntoMemory();
    feed_data_thread_.reset(new std::thread([&]() {
      dataset_->WaitPreLoadDone();
      FeedPass();
    }));
    VLOG(3) << "After PreLoadIntoMemory()";
  }
  void WaitFeedPassDone() { feed_data_thread_->join(); }

 private:
  Dataset* dataset_;
  std::shared_ptr<std::thread> feed_data_thread_;
  int year_;
  int month_;
  int day_;
  // notify boxps to feed this pass feasigns from SSD to memory
  void FeedPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    std::vector<Record> pass_data;
    std::vector<uint64_t> feasign_to_box;
    input_channel_->ReadAll(pass_data);

    auto& index_map = dataset_->GetReaders()[0]->index_omited_in_feedpass_;
    for (const auto& ins : pass_data) {
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        if (index_map.find(feasign.slot()) != index_map.end()) {
          continue;
        }
        feasign_to_box.push_back(feasign.sign().uint64_feasign_);
      }
    }
    input_channel_->Open();
    input_channel_->Write(pass_data);
    input_channel_->Close();
    PADDLEBOX_LOG << "call boxps feedpass";
    //struct std::tm b = {0,0,0,day_,month_ - 1,year_ - 1900}; /* July 5, 2004 */
    struct std::tm b;
    b.tm_year = year_ - 1900;
    b.tm_mon = month_ - 1;
    b.tm_mday = day_;
    b.tm_min = b.tm_hour = b.tm_sec = 0;
    std::time_t x = std::mktime(&b);
    box_ptr->FeedPass(x / 86400, feasign_to_box);
    PADDLEBOX_LOG << "return from boxps feedpass";
  }
};

}  // end namespace framework
}  // end namespace paddle
