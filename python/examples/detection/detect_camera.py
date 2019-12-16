"""Example using TF Lite to detect objects in a laptop camera."""


import argparse
import cv2
import time
import numpy as np

from PIL import Image
from PIL import ImageDraw

import detect
from detect_image import load_labels, make_interpreter, draw_objects
import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

def draw_objects_and_info(draw, objs, labels, info):
  draw_objects(draw, objs, labels)
  draw.text((0,0), info, fill='green')

class MovingAvgPerf():
  def __init__(self, nticks=10):
    self.times = []
    self.nticks = nticks

  def tick(self, diff):
    self.times.append(diff)
    if len(self.times) > self.nticks:
      self.times.pop(0)

  def fps_str(self):
    fps = len(self.times) / sum(self.times)
    return '%.2f fps' % fps

class MovingWindowPerf(MovingAvgPerf):

  def tick(self):
    super().tick(time.monotonic())

  def fps_str(self):
    if len(self.times) == 1:
      fps = 0
    else:
      fps = len(self.times) / (self.times[-1] - self.times[0])
    return '%.2f fps' % fps


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects.')
  parser.add_argument('-o', '--output',
                      help='File path for the result video with annotations (eg: images/processed.avi)')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  video_capture = cv2.VideoCapture(0)
  video_out = None

  perf = MovingWindowPerf()
  tpuperf = MovingAvgPerf()
  print('Running... Press q to quit.')
  while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Crop the central square of the video feed
    height, width, _ = frame.shape   # Get dimensions
    new_width = min(width, height)
    new_height = new_width

    left = int( (width - new_width)/2 )
    top = int( (height - new_height)/2 )
    right = int( (width + new_width)/2 )
    bottom = int( (height + new_height)/2 )

    image = Image.fromarray(frame[top:bottom,left:right])
    if video_out is None and args.output:
      video_out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'), 17, (new_width,new_height))
    
    # Inference
    scale = detect.set_input(interpreter, image.size,
                             lambda size: image.resize(size, Image.ANTIALIAS))
    start = time.monotonic()
    interpreter.invoke()
    tpuperf.tick(time.monotonic() - start)
    perf.tick()
    objs = detect.get_output(interpreter, args.threshold, scale)

    # Display the resulting frame
    msg = '%s\n%s tpu rate' % (perf.fps_str(), tpuperf.fps_str())
    draw_objects_and_info(ImageDraw.Draw(image), objs, labels, msg)
    cv2.imshow('Video', np.asarray(image))

    if video_out is not None:
      video_out.write(np.asarray(image))

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # When everything is done, release the capture
  if video_out is not None:
    video_out.release()
  video_capture.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
