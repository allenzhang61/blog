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
struct DeepseekLayerWeights;

// DeepSeek FFN 子层（每层一个实例）：dense 层用 dense FFN，MoE 层用 MoE FFN。
class MLP : public Module {
public:
    MLP(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool);

    void forward(DeepseekSession &session, float *d_hidden, int input_size);

private:
    const DeepseekLayerWeights &weights_;
    DenseFFN dense_;
    MoE moe_;
};

#endif // LOCAL_LLM_DEEPSEEK_MLP_H
