

int main(void) {
    while(1) {};
}


__attribute__((naked, noreturn)) void _reset(void) {
    extern long _sdata, _edata, _sbss, _ebss, _sidata;

    for (long* bss_el = &_sbss; bss_el < &_ebss; bss_el++) {
        *bss_el = 0;
    }

    for (long *dst_el = &_sdata, *src_el = &_sidata; dst_el < &_edata;) {
        *dst_el = *src_el;
        dst_el++;
        src_el++;
    }

    main();
}

extern void _estack(void);  // Defined in link.ld

// 16 standard and 63 STM32WB55-specific handlers
__attribute__((section(".vectors"))) void (*const tab[16 + 63])(void) = {
  _estack, _reset, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0
};
