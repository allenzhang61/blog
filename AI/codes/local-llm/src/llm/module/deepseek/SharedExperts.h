//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_SHARED_EXPERTS_H
#define LOCAL_LLM_DEEPSEEK_SHARED_EXPERTS_H

#include "llm/module/Module.h"
#include "tensor/Tensor.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class SharedExperts : public Module {
public:
    SharedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    // normed：归一化后的输入 [input_size, hidden_size]；moe：累加输出 [input_size, hidden_size]。
    void forward(DeepseekSession &session, const Tensor &normed, const Tensor &moe);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &lw_;
};

#endif // LOCAL_LLM_DEEPSEEK_SHARED_EXPERTS_H
