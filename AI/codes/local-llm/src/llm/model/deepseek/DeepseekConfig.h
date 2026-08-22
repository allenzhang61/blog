//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEKCONFIG_H
#define LOCAL_LLM_DEEPSEEKCONFIG_H

#include <string>

class MF;

// DeepSeek-V2-Lite 超参，全部从 GGUF 元数据（arch=deepseek2）读取。
// 见 design.md「Ground-Truth GGUF Layout」。
struct DeepseekConfig {
    // 基本结构
    int hidden_size = 2048;
    int num_layers = 27;
    int vocab_size = 102400;
    int num_heads = 16; // number of attention heads，也就是注意力头的数量；在多头注意力里，hidden 会被拆成多个 head

    // MLA
    int kv_lora_rank = 512; // latent 维
    int qk_nope_head_dim = 128; // 每 head 不旋转部分；表示这部分 head dim 不应用位置编码；No Positional Embedding
    // 每 head 旋转部分（解耦 RoPE，k_rope 所有 head 共享）；表示这部分 head dim 会应用 RoPE 旋转位置编码；Rotary Position Embedding
    int qk_rope_head_dim = 64;
    int v_head_dim = 128; // value 每 head 维
    int rope_dim = 64; // = qk_rope_head_dim
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-6f;

    // YARN rope scaling
    bool use_yarn = true;
    float yarn_scaling_factor = 40.0f;
    int yarn_original_context = 4096;
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;
    float yarn_mscale = 0.707f; // mscale_all_dim（V2-Lite）
    float yarn_mscale_base = 0.707f; // 用于 attn softmax scale 的 mscale

    // MoE
    int expert_count = 64;
    int expert_used = 6;
    int expert_shared = 2;
    int first_k_dense = 1; // 前 first_k_dense 层为 dense（V2-Lite=1，即仅 layer 0）
    int dense_ffn = 10944; // dense 层 SwiGLU 中间维
    int expert_ffn = 1408; // 每专家 SwiGLU 中间维
    float routed_scaling = 1.0f; // routed_scaling_factor（V2-Lite=1.0）
    bool norm_topk_prob = false; // V2-Lite=false

    int eos_token_id = -1;
    int bos_token_id = -1;

    // 派生量
    int qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; } // 192
    int shared_ffn() const { return expert_shared * expert_ffn; } // 2816

    // 从模型文件元数据构造。
    explicit DeepseekConfig(const MF &mf);

    void debug_dump() const;
};

#endif // LOCAL_LLM_DEEPSEEKCONFIG_H
