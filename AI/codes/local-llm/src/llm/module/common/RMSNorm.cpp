//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/RMSNorm.h"

#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

void RMSNorm::forward(const StorageTensor &s_weight, const GPUTensor &g_input_f32,
                      const GPUTensor &g_output_f32, float eps, bool one_plus) {
    TensorTool::rms_norm(s_weight, g_input_f32, g_output_f32, eps, one_plus);
}
