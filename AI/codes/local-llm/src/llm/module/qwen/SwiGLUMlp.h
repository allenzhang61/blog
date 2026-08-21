//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_SWIGLUMLP_H
#define LOCAL_LLM_SWIGLUMLP_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/DiskTensor.h"

struct MlpWeights;
class QwenSession;

// SwiGLU MLP 子层：
//   down( SiLU(gate(x)) * up(x) )
//   - gate / up：hidden_size(2560) -> intermediate_size(9216)；
//   - down：intermediate_size(9216) -> hidden_size(2560)。
// 中间量（gate/up/prod 等）走 QwenSession::scratch。
class SwiGLUMlp : public Module {
public:
    SwiGLUMlp(const MlpWeights &weights);

    // 对隐状态做 MLP（prefill 时行数=tokens，decode 时行数=1，由 in.shape 推出）。
    // in / out：[rows, hidden_size]，允许原位。中间量从 session.scratch 取。
    void forward(QwenSession &session, const GPUTensor &in, const GPUTensor &out);

private:
    const MlpWeights &weights_;
};


#endif //LOCAL_LLM_SWIGLUMLP_H
