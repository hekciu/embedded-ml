ENTRY(_reset);

SEARCH_DIR(tflite-micro/gen/cortex_m_generic_cortex-m4_default_cmsis_nn_gcc/lib)
INPUT(-ltensorflow-microlite)


MEMORY {
  flash(rx)  : ORIGIN = 0x08000000, LENGTH = 2048k
  sram(rwx) : ORIGIN = 0x20000000, LENGTH = 192k  /* remaining 64k in a separate address space */
}

_estack     = ORIGIN(sram) + LENGTH(sram);    /* stack points to end of SRAM */

SECTIONS {
    .vectors  : { KEEP(*(.vectors)) }   > flash
    .text     : {
        *(.text)
        tensorflow-microlite:*(.text)
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
