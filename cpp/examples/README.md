# Simple C++ code example

These examples show how to build a simple C++ program that uses the EdgeTPU
runtime library. This subdirectory contains 2 different examples:

- lstpu: 
  - lists available Edge TPU devices (It does not perform inference).
- classification: 
  - classify a bmp image using a classification model.

## Compile the examples
```bash
$ make DOCKER_TARGET=examples DOCKER_CPUS=k8 docker-build
```