#!/bin/bash
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
MODEL_DIR="${SCRIPTPATH}/models"
TEST_DATA_URL=https://github.com/google-coral/edgetpu/raw/master/test_data
mkdir -p "${MODEL_DIR}"
(cd "${MODEL_DIR}"
curl -OL "${TEST_DATA_URL}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite" \
     -OL "${TEST_DATA_URL}/mobilenet_ssd_v2_coco_quant_postprocess.tflite" \
     -OL "${TEST_DATA_URL}/coco_labels.txt")

# Get example image
IMAGE_DIR="${SCRIPTPATH}/images"
mkdir -p "${IMAGE_DIR}"
(cd "${IMAGE_DIR}"
curl -OL "${TEST_DATA_URL}/grace_hopper.bmp")
