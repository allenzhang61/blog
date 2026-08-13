//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_SWIGLUMLP_H
#define LOCAL_LLM_SWIGLUMLP_H

#include <cstddef>

#include "Module.h"

struct MlpWeights;
class QwenForwardScratch;
class CudaWeightPool;

// SwiGLU MLP 子层：
//   down( SiLU(gate(x)) * up(x) )
//   - gate / up：hidden_size(2560) -> intermediate_size(9216)；
//   - down：intermediate_size(9216) -> hidden_size(2560)。
// 中间量（gate/up/prod 等）走 QwenForwardScratch。
class SwiGLUMlp : public Module {
public:
    SwiGLUMlp(const MlpWeights &weights, CudaWeightPool *pool);

    // 对 rows 行隐状态做 MLP（prefill 时 rows=tokens，decode 时 rows=1）。
    // d_in / d_out：[rows, hidden_size]，允许原位。
    void forward(const float *d_in, float *d_out, size_t rows, int hidden_size,
                 QwenForwardScratch &scratch);

private:
    const MlpWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
};


#endif //LOCAL_LLM_SWIGLUMLP_H
