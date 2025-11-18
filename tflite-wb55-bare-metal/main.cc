
#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"



#define MODEL_PATH "/home/hekciu/programming/embedded-ml/sine-wave-model/models/sine_model.tflite"


int main(void) {
    /*
    TfLiteModel* model = TfLiteModelCreateFromFile(MODEL_PATH);
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    
    // Create the interpreter.
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    */

    tflite::InitializeTarget();

    while(1) {};
}

extern "C" {

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

}

