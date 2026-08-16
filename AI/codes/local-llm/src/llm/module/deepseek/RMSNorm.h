//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_RMSNORM_H
#define LOCAL_LLM_DEEPSEEK_RMSNORM_H

#include "format/MF.h"
#include "llm/module/Module.h"

#include <cstddef>

class CudaWeightPool;

namespace deepseek {

// DeepSeek RMSNorm：权重按原值缩放，不使用 Qwen 的 (1 + weight) 约定。
class RMSNorm : public Module {
public:
    RMSNorm(CudaWeightPool *pool, float eps);

    void forward(const MFTensorView &weight, const float *d_in, float *d_out,
                 size_t rows, int hidden_size);

private:
    CudaWeightPool *pool_ = nullptr;
    float eps_ = 1e-6f;
};

} // namespace deepseek

#endif // LOCAL_LLM_DEEPSEEK_RMSNORM_H
