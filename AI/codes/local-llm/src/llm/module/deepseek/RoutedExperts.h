//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_ROUTED_EXPERTS_H
#define LOCAL_LLM_DEEPSEEK_ROUTED_EXPERTS_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"
#include "llm/module/deepseek/MoERouter.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class RoutedExperts : public Module {
public:
    RoutedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    // g_normed：归一化后的输入 [input_size, hidden_size]；g_moe：累加输出 [input_size, hidden_size]。
    void forward(DeepseekSession &session, const GPUTensor &g_normed_f32, const MoERoute &route,
                 const GPUTensor &g_moe_f32);

private:
    const DeepseekConfig &config_;
    // Routed expert 权重：
    //   s_ffn_gate_exps/s_ffn_up_exps [expert_count, expert_ffn, hidden]
    //   s_ffn_down_exps [expert_count, hidden, expert_ffn]
    const DeepseekLayerWeights &lw_;
};

#endif // LOCAL_LLM_DEEPSEEK_ROUTED_EXPERTS_H
