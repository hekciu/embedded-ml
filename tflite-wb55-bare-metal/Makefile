
ARM_GCC_PATH=${PWD}/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-g++
TFLITE_MICRO_PATH=tflite-micro/gen/cortex_m_generic_cortex-m4_default_cmsis_nn_gcc/lib/libtensorflow-microlite.a
TFLITE_INCLUDE_PATH=${PWD}/tflite-micro
FLATBUFFERS_INCLUDE_PATH=${PWD}/tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
GEMMLOWP_INCLUDE_PATH=${PWD}/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp


main.o: main.cc Makefile
	arm-none-eabi-g++ \
    -mcpu=cortex-m4 \
    -fno-exceptions \
    -I${TFLITE_INCLUDE_PATH} \
    -I${FLATBUFFERS_INCLUDE_PATH} \
    -I${GEMMLOWP_INCLUDE_PATH} \
    main.cc -c -o main.o

firmware.elf: main.o
	arm-none-eabi-g++ \
    -T link.ld -nostdlib \
    -L. -l:${TFLITE_MICRO_PATH} \
    main.o -o firmware.elf

firmware.bin: firmware.elf
	arm-none-eabi-objcopy -O binary firmware.elf firmware.bin

firmware.s: main.c
	arm-none-eabi-g++ -mcpu=cortex-m4 main.cc -S -o firmware.s

.PHONY: build flash clean assembly

build: firmware.bin

assembly: firmware.s

flash:
	st-flash --reset write firmware.bin 0x8000000

clean:
	rm -f -- main.o firmware.elf firmware.bin firmware.s
