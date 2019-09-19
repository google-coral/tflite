# Coral examples using TensorFlow Lite API

The example code in this directory uses the [TensorFlow Lite API](
https://www.tensorflow.org/lite) to perform inference on the Edge TPU.

To run this code, you must attach an Edge TPU to your host computer
(or use a device with the Edge TPU built-in such as the Coral Dev Board),
and install the Edge TPU runtime (`libedgetpu.so`) and [`tflite_runtime`](
https://www.tensorflow.org/lite/guide/python).

For Coral device setup instructions, see [g.co/coral/setup](
https://g.co/coral/setup).

Before running the examples, install the required Python dependencies
and download the example models by running the `download.sh` script
located in this directory.
