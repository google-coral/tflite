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
SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
# Allowed CPU values: k8, armv7a, aarch64
CPU ?= k8
# Allowed COMPILATION_MODE values: opt, dbg
COMPILATION_MODE ?= opt

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS := --crosstool_top=@crosstool//:toolchains \
                     --compilation_mode=$(COMPILATION_MODE) \
                     --compiler=gcc \
                     --cpu=$(CPU) \
                     --linkopt=-L$(shell bazel info output_base)/external/edgetpu/libedgetpu/direct/$(CPU) \
                     --linkopt=-l:libedgetpu.so.1

lstpu:
	bazel build $(BAZEL_BUILD_FLAGS) //:lstpu

clean:
	bazel clean
