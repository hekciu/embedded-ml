#!/bin/bash

git clone --depth 1 https://github.com/tensorflow/tflite-micro.git
cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET=cortex_m_generic \
    TARGET_ARCH=cortex-m4 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn microlite \
#    TARGET_TOOLCHAIN_ROOT=/usr/bin/
