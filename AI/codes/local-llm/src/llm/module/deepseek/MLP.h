//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MLP_H
#define LOCAL_LLM_DEEPSEEK_MLP_H

#include "llm/module/Module.h"
#include "llm/module/deepseek/DenseFFN.h"
#include "llm/module/deepseek/MoE.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
class DeepseekWeights;

namespace deepseek {
class RMSNorm;
}

// DeepSeek FFN 子层：前若干层为 dense FFN，后续为 MoE FFN。
class MLP : public Module {
public:
    MLP(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
        deepseek::RMSNorm *rms_norm);

    void forward(DeepseekSession &session, int layer, float *d_hidden, int tokens);

private:
    const DeepseekWeights &weights_;
    DenseFFN dense_;
    MoE moe_;
};

#endif // LOCAL_LLM_DEEPSEEK_MLP_H
