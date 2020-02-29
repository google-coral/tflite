# Image classification example on Coral with TensorFlow Lite

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python
to run an image classification model with acceleration on the Edge TPU, using a
Coral device such as the
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board).

The Python script takes arguments for the model, labels file, and image
you want to process. It then prints the model's prediction for what the
image is to the terminal screen.

## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.withgoogle.com/docs/accelerator/get-started/).

    Importantly, you should have the latest TensorFlow Lite runtime installed,
    as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python).

2.  Clone this Git repo onto your computer:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/tflite --depth 1
    ```

3.  Install this example's dependencies:

    ```
    cd tflite/python/examples/classification

    ./install_requirements.sh
    ```

## Run the code

Use this command to run image classification with the model and photo
downloaded by the above script (photo shown in figure 1):

```
python3 classify_image.py \
  --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
  --labels models/inat_bird_labels.txt \
  --input images/parrot.jpg
```

<img width="200"
     src="https://github.com/google-coral/edgetpu/raw/master/test_data/parrot.jpg" />
<br><b>Figure 1.</b> parrot.jpg

You should see results like this:

```.language-bash
Initializing TF Lite interpreter...
INFO: Initialized TensorFlow Lite runtime.
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
11.8ms
3.0ms
2.8ms
2.9ms
2.9ms
-------RESULTS--------
Ara macao (Scarlet Macaw): 0.76562
```

To demonstrate varying inference speeds, the example repeats the same inference
five times. Your inference speeds might be different based on your host platform
and whether you're using the USB Accelerator with a USB 2.0 or 3.0 connection.

To compare the performance when not using the Edge TPU, try
running it again with the model that's *not* compiled for the Edge TPU:

```
python3 classify_image.py \
  --model models/mobilenet_v2_1.0_224_inat_bird_quant.tflite \
  --labels models/inat_bird_labels.txt \
  --input images/parrot.jpg
```

