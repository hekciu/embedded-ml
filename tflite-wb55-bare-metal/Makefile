
ARM_GCC_PATH=${PWD}/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-g++
TFLITE_MICRO_PATH=tflite-micro/gen/cortex_m_generic_cortex-m4_default_cmsis_nn_gcc/lib/libtensorflow-microlite.a


main.o: main.c
	arm-none-eabi-gcc -mcpu=cortex-m4 main.c -c -o main.o

proj.o: main.o
	arm-none-eabi-gcc -mcpu=cortex-m4 -l:${TFLITE_MICRO_PATH} main.o -o proj.o

firmware.elf: main.o
	arm-none-eabi-gcc -L. -l:${TFLITE_MICRO_PATH} -T link.ld -nostdlib main.o -o firmware.elf

firmware.bin: firmware.elf
	arm-none-eabi-objcopy -O binary firmware.elf firmware.bin

firmware.s: main.c
	arm-none-eabi-gcc -mcpu=cortex-m4 main.c -S -o firmware.s

.PHONY: build flash clean assembly

build: firmware.bin

assembly: firmware.s

flash:
	st-flash --reset write firmware.bin 0x8000000

clean:
	rm -f -- main.o firmware.elf firmware.bin firmware.s
