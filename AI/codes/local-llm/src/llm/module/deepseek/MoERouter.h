//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H
#define LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

struct MoERoute {
    CPUTensor c_expert_ids_i32;
    CPUTensor c_weights_f32;
    // decode（input_size==1）路径：top_w 不回读到 host，直接保留 device 指针供加权累加 kernel 使用。
    // prefill 路径下为 nullptr（沿用 c_weights_f32）。
    const float *d_weights_f32 = nullptr;
    bool decode_device = false;
};

class MoERouter : public Module {
public:
    MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    MoERoute forward(DeepseekSession &session, const GPUTensor &g_normed_f32);

private:
    const DeepseekConfig &config_;
    // Router 权重：s_ffn_gate_inp [expert_count, hidden]，输出每 token 的 top-k expert。
    const DeepseekLayerWeights &lw_;
};

#endif // LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H
