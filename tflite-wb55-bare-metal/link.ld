ENTRY(_reset);

INCLUDE tflite-micro/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/Core/Template/Device_M/Config/Device_gcc.ld

SEARCH_DIR(tflite-micro/gen/cortex_m_generic_cortex-m4_default_cmsis_nn_gcc/lib)
INPUT(-ltensorflow-microlite)

SEARCH_DIR(tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/arm-none-eabi/lib)
INPUT(-lstdc++)
INPUT(-lm)
INPUT(-lc)
INPUT(-lc_nano)
INPUT(-lg)
INPUT(-lgfortran)
INPUT(-lgfortran)
INPUT(-lgloss-linux)
INPUT(-lg_nano)
INPUT(-lnosys)
INPUT(-lrdimon)
INPUT(-lrdimon_nano)
INPUT(-lrdimon-v2m)
INPUT(-lrdpmon)
INPUT(-lstdc++exp)
INPUT(-lstdc++_nano)
INPUT(-lsupc++)
INPUT(-lsupc++_nano)

SEARCH_DIR(tensorflow/lite/micro/tools/make/downloads/gcc_embedded/lib/gcc/arm-none-eabi/14.3.1)
INPUT(-lgcc)


MEMORY {
  flash(rx)  : ORIGIN = 0x08000000, LENGTH = 200k
  sram(rwx) : ORIGIN = 0x20000000, LENGTH = 192k  /* remaining 64k in a separate address space */
}

_estack     = ORIGIN(sram) + LENGTH(sram);    /* stack points to end of SRAM */

SECTIONS {
    .vectors  : { KEEP(*(.vectors)) }   > flash
    .text     : {
        *(.text)
        tensorflow-microlite:*(.text)
        stdc++:*(.text)
        m:*(.text)
        c:*(.text)
        c_nano:*(.text)
        g:*(.text)
        gfortran:*(.text)
        gfortran:*(.text)
        gloss-linux:*(.text)
        g_nano:*(.text)
        nosys:*(.text)
        rdimon:*(.text)
        rdimon_nano:*(.text)
        rdimon-v2m:*(.text)
        rdpmon:*(.text)
        stdc++exp:*(.text)
        stdc++_nano:*(.text)
        supc++:*(.text)
        supc++_nano:*(.text)
        gcc:*(.text)
    }           > flash
    .rodata   : { *(.rodata*) }         > flash

    .data : {
        _sdata = .;   /* .data section start */
        *(.first_data)
        *(.data SORT(.data.*))
        _edata = .;  /* .data section end */
    } > sram AT > flash
    _sidata = LOADADDR(.data);

    .bss : {
    _sbss = .;              /* .bss section start */
    *(.bss SORT(.bss.*) COMMON)
    _ebss = .;              /* .bss section end */
    } > sram

    . = ALIGN(8);
    _end = .;     /* for cmsis_gcc.h  */
}
