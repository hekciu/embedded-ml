/**
  ******************************************************************************
  * @file    basic_model_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-10-25T22:16:48+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef BASIC_MODEL_DATA_PARAMS_H
#define BASIC_MODEL_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_BASIC_MODEL_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_basic_model_data_weights_params[1]))
*/

#define AI_BASIC_MODEL_DATA_CONFIG               (NULL)


#define AI_BASIC_MODEL_DATA_ACTIVATIONS_SIZES \
  { 3648, }
#define AI_BASIC_MODEL_DATA_ACTIVATIONS_SIZE     (3648)
#define AI_BASIC_MODEL_DATA_ACTIVATIONS_COUNT    (1)
#define AI_BASIC_MODEL_DATA_ACTIVATION_1_SIZE    (3648)



#define AI_BASIC_MODEL_DATA_WEIGHTS_SIZES \
  { 407080, }
#define AI_BASIC_MODEL_DATA_WEIGHTS_SIZE         (407080)
#define AI_BASIC_MODEL_DATA_WEIGHTS_COUNT        (1)
#define AI_BASIC_MODEL_DATA_WEIGHT_1_SIZE        (407080)



#define AI_BASIC_MODEL_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_basic_model_activations_table[1])

extern ai_handle g_basic_model_activations_table[1 + 2];



#define AI_BASIC_MODEL_DATA_WEIGHTS_TABLE_GET() \
  (&g_basic_model_weights_table[1])

extern ai_handle g_basic_model_weights_table[1 + 2];


#endif    /* BASIC_MODEL_DATA_PARAMS_H */
