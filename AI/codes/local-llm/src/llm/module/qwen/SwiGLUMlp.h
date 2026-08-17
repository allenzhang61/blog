//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_SWIGLUMLP_H
#define LOCAL_LLM_SWIGLUMLP_H

#include <cstddef>

#include "llm/module/Module.h"

struct MlpWeights;
class QwenSession;
class CudaWeightPool;

// SwiGLU MLP 子层：
//   down( SiLU(gate(x)) * up(x) )
//   - gate / up：hidden_size(2560) -> intermediate_size(9216)；
//   - down：intermediate_size(9216) -> hidden_size(2560)。
// 中间量（gate/up/prod 等）走 QwenSession::scratch。
class SwiGLUMlp : public Module {
public:
    SwiGLUMlp(const MlpWeights &weights, CudaWeightPool *pool);

    // 对 rows 行隐状态做 MLP（prefill 时 rows=tokens，decode 时 rows=1）。
    // d_in / d_out：[rows, hidden_size]，允许原位。中间量从 session.scratch 取。
    void forward(QwenSession &session, const float *d_in, float *d_out, size_t rows, int hidden_size);

private:
    const MlpWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
};


#endif //LOCAL_LLM_SWIGLUMLP_H
