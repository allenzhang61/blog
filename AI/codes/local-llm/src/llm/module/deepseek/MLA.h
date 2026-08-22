//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MLA_H
#define LOCAL_LLM_DEEPSEEK_MLA_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

// MLA attention 子层（每层一个实例，持有该层的权重引用）。
class MLA : public Module {
public:
    MLA(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    // g_hidden：输入/原位更新的隐状态 [input_size, hidden_size]，input_size 由 shape 推出。
    void forward(DeepseekSession &session, const GPUTensor &g_hidden_f32, int start_pos);

private:
    const DeepseekConfig &config_;
    // MLA 权重：
    //   s_attn_norm [hidden]
    //   s_attn_q [num_heads*qk_head_dim, hidden]
    //   s_attn_kv_a_mqa [kv_lora_rank+qk_rope_head_dim, hidden]
    //   s_attn_kv_a_norm [kv_lora_rank]
    //   s_attn_kv_b [num_heads*(qk_nope_head_dim+v_head_dim), kv_lora_rank]
    //   s_attn_output [hidden, num_heads*v_head_dim]
    const DeepseekLayerWeights &lw_;
};

#endif // LOCAL_LLM_DEEPSEEK_MLA_H
