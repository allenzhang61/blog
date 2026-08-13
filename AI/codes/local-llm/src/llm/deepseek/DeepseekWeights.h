//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEKWEIGHTS_H
#define LOCAL_LLM_DEEPSEEKWEIGHTS_H

#include "DeepseekConfig.h"
#include "format/TensorContainer.h"

#include <memory>
#include <string>
#include <vector>

// DeepSeek-V2-Lite 权重：全部来自单个张量容器（GGUF，mmap，零拷贝）。
// 每个张量以 const TensorView* 引用（host mmap 数据 + dtype + shape），
// GPU 上传/反量化在 Module 前向里经 CudaWeightPool 惰性完成。
//
// shape 约定见 TensorContainer：逻辑行主序 [out, in]（GGUF 内部 dims 已在容器层反转）。

struct DeepseekLayerWeights {
    // 归一化
    const TensorView *attn_norm = nullptr;  // blk.i.attn_norm.weight [hidden]
    const TensorView *ffn_norm = nullptr;   // blk.i.ffn_norm.weight  [hidden]

    // MLA
    const TensorView *attn_q = nullptr;         // [hidden, n_heads*qk_head]  (in=hidden,out=3072)
    const TensorView *attn_kv_a_mqa = nullptr;  // [hidden, kv_lora+qk_rope]  (out=576)
    const TensorView *attn_kv_a_norm = nullptr; // [kv_lora]
    const TensorView *attn_kv_b = nullptr;      // [kv_lora, n_heads*(qk_nope+v_head)] (out=4096)
    const TensorView *attn_output = nullptr;    // [n_heads*v_head, hidden] (in=2048,out=2048)

    bool is_moe = false;

    // dense 层（layer < first_k_dense）
    const TensorView *ffn_gate = nullptr; // [hidden, dense_ffn]
    const TensorView *ffn_up = nullptr;
    const TensorView *ffn_down = nullptr; // [dense_ffn, hidden]

    // MoE 层
    const TensorView *ffn_gate_inp = nullptr;    // router [hidden, n_experts] F32
    const TensorView *ffn_gate_exps = nullptr;   // [hidden, expert_ffn, n_experts]
    const TensorView *ffn_up_exps = nullptr;     // [hidden, expert_ffn, n_experts]
    const TensorView *ffn_down_exps = nullptr;   // [expert_ffn, hidden, n_experts]
    const TensorView *ffn_gate_shexp = nullptr;  // shared [hidden, shared_ffn]
    const TensorView *ffn_up_shexp = nullptr;
    const TensorView *ffn_down_shexp = nullptr;  // [shared_ffn, hidden]
};

class DeepseekWeights {
public:
    DeepseekWeights(const TensorContainer &tensors, const DeepseekConfig &config);

    // 顶层
    const TensorView *token_embd = nullptr; // [hidden, vocab]
    const TensorView *output_norm = nullptr; // [hidden]
    const TensorView *output = nullptr;      // lm_head [hidden, vocab]（非 tie）

    std::vector<DeepseekLayerWeights> layers;

};

#endif // LOCAL_LLM_DEEPSEEKWEIGHTS_H
