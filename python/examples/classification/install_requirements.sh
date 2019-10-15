#!/bin/bash
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install required Python packages,
# but not on Mendel (Dev Board)—it has these already and shouldn't use pip
if [[ ! -f /etc/mendel_version ]]; then
  python3 -m pip install numpy Pillow
fi

# Get TF Lite model and labels
MODEL_DIR="${SCRIPTPATH}/models"
TEST_DATA_URL=https://github.com/google-coral/edgetpu/raw/master/test_data
mkdir -p "${MODEL_DIR}"

(cd "${MODEL_DIR}" && \
curl -OL "${TEST_DATA_URL}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
     -OL "${TEST_DATA_URL}/mobilenet_v2_1.0_224_inat_bird_quant.tflite" \
     -OL "${TEST_DATA_URL}/inat_bird_labels.txt")

# Get example image
IMAGE_DIR="${SCRIPTPATH}/images"
mkdir -p "${IMAGE_DIR}"

(cd "${IMAGE_DIR}" && \
curl -OL "${TEST_DATA_URL}/parrot.jpg")
