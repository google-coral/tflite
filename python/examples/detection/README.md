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

Use this command to run object detection with the model and photo
downloaded by the above script (photo shown in figure 1):

```
python3 detect_image.py \
  --model models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels models/coco_labels.txt \
  --input images/grace_hopper.bmp
```

<figure style="margin-left:0">
  <img style="width:200px"
       src="https://github.com/google-coral/edgetpu/raw/master/test_data/grace_hopper.bmp" />
  <figcaption><b>Figure 1.</b> grace_hopper.bmp</figcaption>
</figure>

You should see results like this:

```
INFO: Initialized TensorFlow Lite runtime.
----INFERENCE TIME----
Note: The first inference is slow because it includes loading the model into Edge TPU memory.
33.92 ms
19.71 ms
19.91 ms
19.91 ms
19.90 ms
-------RESULTS--------
-----------------------------------------
person
score =  0.789062
bounding box = 0 513 16 596
-----------------------------------------
tie
score =  0.789062
bounding box = 227 290 425 544
Image saved as object_detection_result.jpg
```

To demonstrate varying inference speeds, the example repeats the same inference
five times. Your inference speeds might be different based on your host platform
and whether you're using the USB Accelerator with a USB 2.0 or 3.0 connection.

To compare the performance when not using the Edge TPU, try
running it again with the model that's *not* compiled for the Edge TPU:

```
python3 detect_image.py \
  --model models/mobilenet_ssd_v2_coco_quant_postprocess.tflite \
  --labels models/coco_labels.txt \
  --input images/grace_hopper.bmp
```
