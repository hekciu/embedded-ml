#define TF_LITE_STATIC_MEMORY

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_allocator.h"
// #include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tflite-micro/tensorflow/lite/micro/micro_interpreter_graph.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tflite-micro/tensorflow/lite/micro/memory_helpers.h"
#include "tflite-micro/tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"


#include "sine_model.cc"
// #include "sine_model_stolen.h"


#include "uart.hpp"
#include "led.hpp"


static void spin(uint32_t ticks) { while (ticks > 0) ticks--; };


const void* sine_model_data = (const void *)_home_hekciu_programming_embedded_ml_tflite_wb55_bare_metal____sine_wave_model_models_sine_model_tflite;
const uint32_t sine_model_size = _home_hekciu_programming_embedded_ml_tflite_wb55_bare_metal____sine_wave_model_models_sine_model_tflite_len;

//const void* sine_model_data = sine_model;
//const uint32_t sine_model_size = sine_model_len;


static constexpr int kTensorArenaSize = 40000;
static uint8_t tensor_arena[kTensorArenaSize];


namespace {
    using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

    TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
      TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
      return kTfLiteOk;
    }
}


typedef struct test {
    char dupa[20];
    // char chuj[100];
} test;

int main(void) {
    tflite::InitializeTarget();

    test t = {};

    return 0;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.

    HelloWorldOpResolver op_resolver;
    TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

    // Arena size just a round number. The exact arena usage can be determined
    // using the RecordingMicroInterpreter.

    // this part work
    tflite::SingleArenaBufferAllocator* memory_allocator =
       tflite::SingleArenaBufferAllocator::Create(tensor_arena, kTensorArenaSize);

    uint8_t* aligned_arena =
          tflite::AlignPointerUp(tensor_arena, tflite::MicroArenaBufferAlignment());

    uint8_t* aligned_arena_AAAAAAAAAAA =
          tflite::AlignPointerDown(tensor_arena, tflite::MicroArenaBufferAlignment());

    //tflite::MicroMemoryPlanner* memory_planner =
    //  tflite::CreateMemoryPlanner(memory_planner_type, memory_allocator);
    
    tflite::MicroMemoryPlanner* memory_planner = nullptr;
    uint8_t* memory_planner_buffer = nullptr;
    /*

    memory_planner_buffer = memory_allocator->AllocatePersistentBuffer(
      sizeof(tflite::LinearMemoryPlanner), alignof(tflite::LinearMemoryPlanner));

    memory_planner = new (memory_planner_buffer) tflite::LinearMemoryPlanner();

    uint8_t* allocator_buffer = memory_allocator->AllocatePersistentBuffer(
      sizeof(tflite::MicroAllocator), alignof(tflite::MicroAllocator));

    tflite::MicroAllocator* allocator = new (allocator_buffer)
      tflite::MicroAllocator(memory_allocator, memory_allocator, memory_planner);

    auto micro_allocator = tflite::MicroAllocator::Create(
              tensor_arena, kTensorArenaSize,
              tflite::MemoryPlannerType::kLinear);
    */

    const tflite::Model* model =
      ::tflite::GetModel(sine_model_data);

    //TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

    TfLiteContext context = {};

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                           kTensorArenaSize);


    TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

    TfLiteTensor* input = interpreter.input(0);
    TFLITE_CHECK_NE(input, nullptr);

    TfLiteTensor* output = interpreter.output(0);
    TFLITE_CHECK_NE(output, nullptr);


    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;

    // Check if the predicted output is within a small range of the
    // expected output
    float epsilon = 0.05;

    constexpr int kNumTestValues = 4;
    float golden_inputs_float[kNumTestValues] = {0.77, 1.57, 2.3, 3.14};

    // The int8 values are calculated using the following formula
    // (golden_inputs_float[i] / input->params.scale + input->params.zero_point)
    int8_t golden_inputs_int8[kNumTestValues] = {-96, -63, -34, 0};

    int8_t output_values[kNumTestValues] = {};

    for (int i = 0; i < kNumTestValues; ++i) {
        input->data.int8[0] = golden_inputs_int8[i];
        TF_LITE_ENSURE_STATUS(interpreter.Invoke());
        // float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
        // TFLITE_CHECK_LE(abs(sin(golden_inputs_float[i]) - y_pred), epsilon);
        output_values[i] = output->data.int8[i];
    }

    return kTfLiteOk;
}


//__attribute__((naked, noreturn)) void _reset(void) {
extern "C" __attribute__((naked, noreturn)) void Reset_Handler(void) {
    extern long _sdata, _edata, _sbss, _ebss, _sidata;

    for (long* bss_el = &_sbss; bss_el < &_ebss; bss_el++) {
        *bss_el = 0;
    }

    for (long *dst_el = &_sdata, *src_el = &_sidata; dst_el < &_edata;) {
        *dst_el = *src_el;
        dst_el++;
        src_el++;
    }

    /* This one does work */
    setup_green_led();
    uart_init(115200);

    /* This one does not */
    // setup_green_led();
    // uart_init(115200);

    for(;;) {
        // main();

        blink_green_led();

        uart_transmit("dupa dupa\r\n");

        spin(99999);
    }
}


extern "C" void _estack(void);  // Defined in link.ld

// 16 standard and 63 STM32WB55-specific handlers
__attribute__((section(".vectors"))) void (*const tab[16 + 63])(void) = {
  _estack, Reset_Handler, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0
};


