//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MLA_H
#define LOCAL_LLM_DEEPSEEK_MLA_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/DiskTensor.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

// MLA attention 子层（每层一个实例，持有该层的权重引用）。
class MLA : public Module {
public:
    MLA(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    // hidden：输入/原位更新的隐状态 [input_size, hidden_size]，input_size 由 shape 推出。
    void forward(DeepseekSession &session, const GPUTensor &hidden, int start_pos);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &lw_;
};

#endif // LOCAL_LLM_DEEPSEEK_MLA_H
