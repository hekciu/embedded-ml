#!/bin/bash

CMSIS_DIR=cmsis/

git clone --depth 1 -b 5.9.0 https://github.com/ARM-software/CMSIS_5 $CMSIS_DIR/cmsis_core
git clone --depth 1 -b master https://github.com/STMicroelectronics/cmsis_device_wb $CMSIS_DIR/cmsis_wb

git clone --depth 1 https://github.com/tensorflow/tflite-micro.git
cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET=cortex_m_generic \
    TARGET_ARCH=cortex-m4 \
    OPTIMIZED_KERNEL_DIR=cmsis_nn \
    microlite

#    TARGET_TOOLCHAIN_ROOT=/usr/bin/ \

