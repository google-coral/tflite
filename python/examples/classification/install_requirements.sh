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
  if ! python3 -m pip --version > /dev/null; then
    echo "Install pip first by following https://pip.pypa.io/en/stable/installing/ guide."
    exit 1
  fi
  python3 -m pip install numpy Pillow
fi

# Get TF Lite model and labels
MODEL_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODEL_DIR}"

(cd "${MODEL_DIR}" && \
curl -OL "${TEST_DATA_URL}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
     -OL "${TEST_DATA_URL}/mobilenet_v2_1.0_224_inat_bird_quant.tflite" \
     -OL "${TEST_DATA_URL}/inat_bird_labels.txt")

# Get example image
IMAGE_DIR="${SCRIPT_DIR}/images"
mkdir -p "${IMAGE_DIR}"

(cd "${IMAGE_DIR}" && \
curl -OL "${TEST_DATA_URL}/parrot.jpg")
