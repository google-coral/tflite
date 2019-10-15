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
"""Functions to work with detection models."""

import collections
import numpy as np

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
  __slots__ = ()

  @property
  def width(self):
    return self.xmax - self.xmin

  @property
  def height(self):
    return self.ymax - self.ymin

  @property
  def area(self):
    return self.width * self.height

  @property
  def valid(self):
    return self.width >= 0 and self.height >= 0

  @staticmethod
  def intersect(a, b):
    return BBox(xmin=max(a.xmin, b.xmin),
                ymin=max(a.ymin, b.ymin),
                xmax=min(a.xmax, b.xmax),
                ymax=min(a.ymax, b.ymax))

  @staticmethod
  def union(a, b):
    return BBox(xmin=min(a.xmin, b.xmin),
                ymin=min(a.ymin, b.ymin),
                xmax=max(a.xmax, b.xmax),
                ymax=max(a.ymax, b.ymax))

  @staticmethod
  def intersection_over_union(a, b):
    intersection = BBox.intersect(a, b)
    if not intersection.valid:
      return 0.0
    intersection_area = intersection.area
    return intersection_area / (a.area + b.area - intersection_area)


def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  return width, height


def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, size, resize):
  """Copies a resized and properly zero-padded image to the input tensor.

  Args:
    interpreter: Interpreter object.
    size: original image size as (width, height) tuple.
    resize: a function that takes a (width, height) tuple, and returns an RGB
      image resized to those dimensions.
  Returns:
    Actual resize ratio, which should be passed to `get_output` function.
  """
  width, height = input_size(interpreter)
  w, h = size
  scale = min(width / w, height / h)
  w, h = int(w * scale), int(h * scale)
  tensor = input_tensor(interpreter)
  tensor.fill(0)  # padding
  tensor[:h, :w] = resize((w, h))
  return scale


def output_tensor(interpreter, i):
  """Returns output tensor view."""
  tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
  return np.squeeze(tensor)


def get_output(interpreter, score_threshold, image_scale=1.0):
  """Returns list of detected objects."""
  boxes = output_tensor(interpreter, 0)
  class_ids = output_tensor(interpreter, 1)
  scores = output_tensor(interpreter, 2)
  count = int(output_tensor(interpreter, 3))

  width, height = input_size(interpreter)
  scale_x, scale_y = width / image_scale, height / image_scale

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    xmin, xmax = int(scale_x * xmin), int(scale_x * xmax)
    ymin, ymax = int(scale_y * ymin), int(scale_y * ymax)
    return Object(
        id=int(class_ids[i]),
        score=scores[i],
        bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

  return [make(i) for i in range(count) if scores[i] >= score_threshold]
