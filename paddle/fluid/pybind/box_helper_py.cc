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
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/pybind/box_helper_py.h"
#ifdef PADDLE_WITH_BOX_PS
#include <boxps_public.h>
#endif

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindBoxHelper(py::module* m) {
  py::class_<framework::BoxHelper, std::shared_ptr<framework::BoxHelper>>(
      *m, "BoxPS")
      .def(py::init([](paddle::framework::Dataset* dataset, int year, int month, int day) {
        return std::make_shared<paddle::framework::BoxHelper>(dataset, year, month, day);
      }))
      .def("begin_pass", &framework::BoxHelper::BeginPass, py::call_guard<py::gil_scoped_release>())
      .def("end_pass", &framework::BoxHelper::EndPass, py::call_guard<py::gil_scoped_release>())
      .def("wait_feed_pass_done", &framework::BoxHelper::WaitFeedPassDone, py::call_guard<py::gil_scoped_release>())
      .def("preload_into_memory", &framework::BoxHelper::PreLoadIntoMemory, py::call_guard<py::gil_scoped_release>())
      .def("load_into_memory", &framework::BoxHelper::LoadIntoMemory, py::call_guard<py::gil_scoped_release>());
}  // end BoxHelper

void BindBoxWrapper(py::module* m) {
  py::class_<framework::BoxWrapper, std::shared_ptr<framework::BoxWrapper>>(
      *m, "BoxWrapper")
      .def(py::init([]() {
        // return std::make_shared<paddle::framework::BoxHelper>(dataset);
        return framework::BoxWrapper::GetInstance();
      }))
      .def("save_base", &framework::BoxWrapper::SaveBase)
      .def("save_delta", &framework::BoxWrapper::SaveDelta)
      .def("initialize_gpu", &framework::BoxWrapper::InitializeGPU)
      .def("finalize", &framework::BoxWrapper::Finalize);
}  // end BoxWrapper

void BindSaveModelStat(py::module* m) {
  py::class_<boxps::SaveModelStat, std::shared_ptr<boxps::SaveModelStat>>(
      *m, "SaveModelStat")
      .def(py::init())
      .def_readwrite("total_key_count", &boxps::SaveModelStat::total_key_count)
      .def_readwrite("total_embedx_key_count", &boxps::SaveModelStat::total_embedx_key_count)
      .def_readwrite("xbox_key_count", &boxps::SaveModelStat::xbox_key_count)
      .def_readwrite("xbox_embedx_key_count", &boxps::SaveModelStat::xbox_embedx_key_count)
      .def_readwrite("ctr_key_count", &boxps::SaveModelStat::ctr_key_count)
      .def_readwrite("ctr_embedx_key_count", &boxps::SaveModelStat::ctr_embedx_key_count)
      .def_readwrite("ubm_key_count", &boxps::SaveModelStat::ubm_key_count)
      .def_readwrite("ubm_embedx_key_count", &boxps::SaveModelStat::ubm_embedx_key_count)
      .def_readwrite("shrink_key_count", &boxps::SaveModelStat::shrink_key_count)
      .def_readwrite("filter_key_count", &boxps::SaveModelStat::filter_key_count)
      .def_readwrite("invalid_key_count", &boxps::SaveModelStat::invalid_key_count);
}  // end BindSaveModelStat

}  // end namespace pybind
}  // end namespace paddle
