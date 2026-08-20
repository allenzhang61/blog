//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/RMSNorm.h"

void RMSNorm::forward(const Tensor &weight, const Tensor &input, const Tensor &output,
                      float eps, bool one_plus) {
    weight.to_gpu();
    weight.rms_norm(input, output, eps, one_plus);
}
