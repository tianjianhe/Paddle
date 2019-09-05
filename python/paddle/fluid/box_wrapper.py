#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import core

__all__ = ['BoxWrapper']


class BoxWrapper(object):
    """
    box wrapper class
    """

    def __init__(self):
        self.box_wrapper = core.BoxWrapper()

    def save_model(self):
        self.box_wrapper.save_model()

    def initialize_gpu(self, conf_file):
        if not isinstance(conf_file, str):
            raise TypeError(
                "conf_file in parameter of initialize_gpu should be str")
        self.box_wrapper.initialize_gpu(conf_file)

    def finalize(self):
        self.box_wrapper.finalize()
