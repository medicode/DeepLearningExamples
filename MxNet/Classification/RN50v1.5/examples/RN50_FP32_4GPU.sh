# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


# This script launches ResNet50 training in FP32 on 4 GPUs using 384 batch size (96 per GPU)
# Usage ./RN50_FP32_4GPU.sh <path to this repository> <additionals flags>

"$1/runner" -n 4 -b 96 --dtype float32 --model-prefix model ${@:2}
