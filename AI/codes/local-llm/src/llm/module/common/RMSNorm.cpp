//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/RMSNorm.h"

#include "tensor/TensorTool.h"

void RMSNorm::forward(const Tensor &weight, const Tensor &input, const Tensor &output,
                      float eps, bool one_plus) {
    TensorTool::rms_norm(weight, input, output, eps, one_plus);
}
