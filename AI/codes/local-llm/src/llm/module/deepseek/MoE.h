//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MOE_H
#define LOCAL_LLM_DEEPSEEK_MOE_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"
#include "llm/module/deepseek/MoERouter.h"
#include "llm/module/deepseek/RoutedExperts.h"
#include "llm/module/deepseek/SharedExperts.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class MoE : public Module {
public:
    MoE(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    void forward(DeepseekSession &session, const GPUTensor &g_hidden);

private:
    const DeepseekConfig &config_;
    // MoE FFN 权重：
    //   s_ffn_norm [hidden]
    //   s_ffn_gate_inp [expert_count, hidden]
    //   s_ffn_gate_exps/s_ffn_up_exps [expert_count, expert_ffn, hidden]
    //   s_ffn_down_exps [expert_count, hidden, expert_ffn]
    //   s_ffn_gate_shexp/s_ffn_up_shexp [shared_ffn, hidden]
    //   s_ffn_down_shexp [hidden, shared_ffn]
    const DeepseekLayerWeights &weights_;
    MoERouter router_;
    RoutedExperts routed_experts_;
    SharedExperts shared_experts_;
};

#endif // LOCAL_LLM_DEEPSEEK_MOE_H
