# This is a Makefile to cross-compile classify.cc example.
# 1. Download latest Edge TPU runtime archive from https://coral.ai/software/
#    and extract next to the Makefile:
#    $ wget https://dl.google.com/coral/edgetpu_api/edgetpu_runtime_20200710.zip
#    $ unzip edgetpu_runtime_20200710.zip
# 2. Download TensorFlow to the Linux machine:
#    $ git clone https://github.com/tensorflow/tensorflow.git
# 3. Download external dependencies for TensorFlow Lite:
#    $ tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
# 4. Cross-compile TensorFlow Lite for aarch64:
#    $ tensorflow/tensorflow/lite/tools/make/build_generic_aarch64_lib.sh
# 5. Cross-compile classify.cc example for aarch64:
#    $ TENSORFLOW_DIR=<location> make
# 6. Copy the following files to Coral Dev board:
#      * Generated 'classify' binary
#      * Model file 'mobilenet_v1_1.0_224_quant_edgetpu.tflite' from 'test_data'
#      * Label file 'imagenet_labels.txt ' from 'test_data'
#      * Image file 'resized_cat.bmp' from 'test_data'
#    and finally run 'classify' binary on the board:
#    $ classify mobilenet_v1_1.0_224_quant_edgetpu.tflite \
#               imagenet_labels.txt \
#               resized_cat.bmp \
#               0.0001
#    INFO: Initialized TensorFlow Lite runtime.
#    INFO: Replacing 1 node(s) with delegate (EdgeTpuDelegateForCustomOp) node, yielding 1 partitions.
#    0.81641 286  Egyptian cat
#    0.10938 283  tiger cat
#    0.03516 282  tabby, tabby cat
#    0.01172 812  space heater
#    0.00781 754  radiator
#    0.00391 540  doormat, welcome mat
#    0.00391 285  Siamese cat, Siamese
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TENSORFLOW_DIR ?=

classify: classify.cc
	aarch64-linux-gnu-g++ -std=c++11 -o classify classify.cc \
	-I$(MAKEFILE_DIR)/edgetpu_runtime/libedgetpu/ \
	-I$(TENSORFLOW_DIR) \
	-I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-L$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib \
	-L$(MAKEFILE_DIR)/edgetpu_runtime/libedgetpu/direct/aarch64/ \
	-ltensorflow-lite -l:libedgetpu.so.1.0 -lpthread -lm -ldl

clean:
	rm -f classify
