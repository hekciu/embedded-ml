#!/bin/bash


MODEL_PATH=${PWD}/../sine-wave-model/models/sine_model.tflite
MODEL_OUTPUT_PATH=${PWD}/sine_model.cc

xxd -i ${MODEL_PATH} > ${MODEL_OUTPUT_PATH}
