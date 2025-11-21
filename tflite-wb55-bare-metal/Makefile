
# The base path for downloaded arm gcc toolchain
# ARM_NONE_EABI=${PWD}/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi

### OLDER VERSION INSTALLED ###
ARM_NONE_EABI=arm-none-eabi

TFLITE_INCLUDE_PATH=${PWD}/tflite-micro
ARM_NONE_EABI_INCLUDE_PATH=tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/arm-none-eabi/include
FLATBUFFERS_INCLUDE_PATH=${PWD}/tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
GEMMLOWP_INCLUDE_PATH=${PWD}/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp


main.o: main.cc Makefile 
	${ARM_NONE_EABI}-g++ \
    -mcpu=cortex-m4 \
    -fno-exceptions \
    -I${TFLITE_INCLUDE_PATH} \
    -I${FLATBUFFERS_INCLUDE_PATH} \
    -I${GEMMLOWP_INCLUDE_PATH} \
    main.cc -c -o main.o

# removed -nostdlib
firmware.elf: main.o
	${ARM_NONE_EABI}-g++ \
	-T link.ld \
    -nostdlib \
    -fno-exceptions \
	main.o -o firmware.elf

firmware.bin: firmware.elf
	${ARM_NONE_EABI}-objcopy -O binary firmware.elf firmware.bin

firmware.s: main.c
	${ARM_NONE_EABI}-g++ -mcpu=cortex-m4 main.cc -S -o firmware.s

.PHONY: build flash clean assembly

build: firmware.bin

assembly: firmware.s

flash:
	st-flash --reset write firmware.bin 0x8000000

clean:
	rm -f -- main.o firmware.elf firmware.bin firmware.s
