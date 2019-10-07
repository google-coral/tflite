# python3
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
r"""Example using TF Lite to classify a given image using an Edge TPU.

   To run this code, you must attach an Edge TPU attached to the host and
   install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
   device setup instructions, see g.co/coral/setup.

   Example usage (use `install_requirements.sh` to get these files):
   ```
   python3 classify_image.py \
     --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
     --labels models/inat_bird_labels.txt \
     --image images/parrot.jpg
   ```
"""

import argparse
import time
import numpy as np

from PIL import Image

import tflite_runtime.interpreter as tflite


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k):
  """Performs image classification.

  Args:
    interpreter: The TF Lite interpreter object.
    image: The image to classify, already downscaled to match the input tensor.
    top_k: The number of classifications to return.

  Returns:
    A list of results sorted by probability, each one as a tuple of
    (label_index, probability).
  """
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered_indices = output.argsort()[-top_k:][::-1]
  return [(i, output[i]) for i in ordered_indices]


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument('--image', help='Image to be classified.', required=True)
  parser.add_argument(
      '--top_k', help='Number of classifications to list', type=int, default=1)
  parser.add_argument(
      '--count', help='Number of times to run inference', type=int, default=5)
  args = parser.parse_args()

  print('Initializing TF Lite interpreter...')
  interpreter = tflite.Interpreter(
      model_path=args.model,
      experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  image = Image.open(args.image).resize((width, height), Image.ANTIALIAS)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    start_time = time.monotonic()
    results = classify_image(interpreter, image, args.top_k)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    print('%.1fms' % elapsed_ms)

  labels = load_labels(args.labels)

  print('-------RESULTS--------')
  for label_id, prob in results:
    print('%s: %.5f' % (labels[label_id], prob))


if __name__ == '__main__':
  main()
