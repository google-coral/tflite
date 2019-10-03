#!/bin/bash
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install required packages
python3 -m pip install -r "${SCRIPTPATH}/requirements.txt"

# Get TF Lite model and labels
MODEL_DIR="${SCRIPTPATH}/models"
mkdir -p "${MODEL_DIR}"

(cd "${MODEL_DIR}" && \
curl -O https://dl.google.com/coral/canned_models/mobilenet_v1_1.0_224_quant_edgetpu.tflite && \
curl -O https://dl.google.com/coral/canned_models/imagenet_labels.txt)
