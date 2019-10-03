# Object detection example on Coral with TensorFlow Lite

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python
to run an object detection model with acceleration on the Edge TPU, using a
Coral device such as the
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board).

The Python script takes arguments for the model, labels file, and image
you want to process. It then prints each detected object and the location
coordinates, and saves/displays the original image with bounding boxes and
labels drawn on top.

## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.withgoogle.com/docs/accelerator/get-started/).

    Importantly, you should have the latest TensorFlow lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)

2.  Clone this Git repo onto your computer:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/tflite --depth 1
    ```

3.  Install this example's dependencies:

    ```
    cd tflite/python/examples/object_detection

    bash install_requirements.sh
    ```

## Run the code

This command uses the model and image provided by the download above:

```
python3 detect_image.py \
  --model models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels models/coco_labels.txt \
  --input images/grace_hopper.bmp
```

To compare the performance when not using the Edge TPU, try
running it again with the model that's *not* compiled for the Edge TPU:

```
python3 detect_image.py \
  --model models/mobilenet_ssd_v2_coco_quant_postprocess.tflite \
  --labels models/coco_labels.txt \
  --input images/grace_hopper.bmp
```
