#!/bin/bash
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_DATA_URL=https://github.com/google-coral/edgetpu/raw/master/test_data

# Install required Python packages,
# but not on Mendel (Dev Board)—it has these already and shouldn't use pip
if [[ ! -f /etc/mendel_version ]]; then
  python3 -m pip install numpy Pillow
fi

# If running Raspberry Pi, also install 'imagemagick' to display images
MODEL=$(tr -d '\0' < /proc/device-tree/model)
if [[ "${MODEL}" == "Raspberry Pi"* ]]; then
  sudo apt-get install imagemagick
fi

# Get TF Lite model and labels
MODEL_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODEL_DIR}"
(cd "${MODEL_DIR}"
curl -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" \
     -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_coco_quant_postprocess.tflite" \
     -OL "${TEST_DATA_URL}/coco_labels.txt")

# Get example image
IMAGE_DIR="${SCRIPT_DIR}/images"
mkdir -p "${IMAGE_DIR}"
(cd "${IMAGE_DIR}"
curl -OL "${TEST_DATA_URL}/grace_hopper.bmp")
