#!/usr/bin/python3
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
"""Example using TF Lite to detect objects in a given image."""

import argparse
import re
import time

import numpy as np

from PIL import Image
from PIL import ImageDraw
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def annotate_objects(results, labels, img):
  """Draws the bounding box and label for each object on the given PIL Image."""
  draw = ImageDraw.Draw(img)

  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * img.width)
    xmax = int(xmax * img.width)
    ymin = int(ymin * img.height)
    ymax = int(ymax * img.height)

    # Print results to screen
    print('-----------------------------------------')
    print(labels[obj['class_id']])
    print('score = ', obj['score'])
    print('bounding box =', xmin, xmax, ymin, ymax)

    # Overlay the box, label, and score on the image
    offset = 10
    draw.rectangle((xmin, ymin, xmax, ymax), outline='red')
    draw.text((xmin + offset, ymin + offset),
              '%s\n%.2f' % (labels[obj['class_id']], obj['score']),
              fill='red')


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--input', help='File path of image to process.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      type=float,
      default=0.4)
  parser.add_argument(
      '--output',
      help='File path for the result image with annotations')
  parser.add_argument(
      '--count', help='Number of times to run inference', type=int, default=5)
  args = parser.parse_args()

  labels = load_labels(args.labels)
  interpreter = Interpreter(
      args.model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  input_image = Image.open(args.input).convert('RGB').resize(
      (input_width, input_height), Image.ANTIALIAS)
  output_image = args.output or 'object_detection_result.jpg'

  print('----INFERENCE TIME----')
  print('Note: The first inference is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    start_time = time.monotonic()
    results = detect_objects(interpreter, input_image, args.threshold)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    print('%.2f ms' % elapsed_ms)

  if not results:
    print('No objects detected')
    return

  # Draw the results on the image and save the file
  print('-------RESULTS--------')
  with Image.open(args.input) as img:
    annotate_objects(results, labels, img)
    img.save(output_image)
    print('Image saved as', output_image)
    # Display it on attached monitor
    img.show()

if __name__ == '__main__':
  main()
