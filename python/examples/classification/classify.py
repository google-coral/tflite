# Lint as: python3
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
"""Functions to work with classification models."""

import collections
import operator
import numpy as np

Class = collections.namedtuple('Class', ['id', 'score'])


def input_details(interpreter, key):
  """Returns input details by specified key."""
  return interpreter.get_input_details()[0][key]


def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = input_details(interpreter, 'shape')
  return width, height


def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = input_details(interpreter, 'index')
  return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter, dequantize=True):
  """Returns output tensor of classification model.

  Integer output tensor is dequantized by default.

  Args:
    interpreter: tflite.Interpreter;
    dequantize: bool; whether to dequantize integer output tensor.

  Returns:
    Output tensor as numpy array.
  """
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())

  if dequantize and np.issubdtype(output_details['dtype'], np.integer):
    scale, zero_point = output_details['quantization']
    return scale * (output_data - zero_point)

  return output_data


def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data


def get_output(interpreter, top_k=1, score_threshold=0.0):
  """Returns no more than top_k classes with score >= score_threshold."""
  scores = output_tensor(interpreter)
  classes = [
      Class(i, scores[i])
      for i in np.argpartition(scores, -top_k)[-top_k:]
      if scores[i] >= score_threshold
  ]
  return sorted(classes, key=operator.itemgetter(1), reverse=True)
