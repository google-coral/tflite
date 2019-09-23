#!/bin/bash
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install required packages
python3 -m pip install -r "${SCRIPTPATH}/requirements.txt"

# Get TF Lite model and labels
MODEL_DIR="${SCRIPTPATH}/models"
mkdir -p "${MODEL_DIR}"

(cd "${MODEL_DIR}" && \
curl -OL https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite && \
curl -OL https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite && \
curl -OL https://github.com/google-coral/edgetpu/raw/master/test_data/inat_bird_labels.txt)

# Get example image
IMAGE_DIR="${SCRIPTPATH}/images"
mkdir -p "${IMAGE_DIR}"

(cd "${IMAGE_DIR}" && \
curl -OL https://github.com/google-coral/edgetpu/raw/master/test_data/parrot.jpg)
