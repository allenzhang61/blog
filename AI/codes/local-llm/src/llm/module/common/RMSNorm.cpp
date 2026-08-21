//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/RMSNorm.h"

#include "tensor/TensorTool.h"

void RMSNorm::forward(const DiskTensor &weight, const GPUTensor &input, const GPUTensor &output,
                      float eps, bool one_plus) {
    TensorTool::rms_norm(weight, input, output, eps, one_plus);
}
