//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEKWEIGHTS_H
#define LOCAL_LLM_DEEPSEEKWEIGHTS_H

#include "DeepseekConfig.h"
#include "format/MF.h"

#include <memory>
#include <string>
#include <vector>

// DeepSeek-V2-Lite 权重：全部来自单个模型文件（GGUF，mmap，零拷贝）。
// 每个张量以 const TensorView* 引用（host mmap 数据 + dtype + shape），
// GPU 上传/反量化在 Module 前向里经 CudaWeightPool 惰性完成。
//
// shape 约定见 MF：逻辑行主序 [out, in]（GGUF 内部 dims 已在容器层反转）。

struct DeepseekLayerWeights {
    // 归一化
    const MFTensorView *attn_norm = nullptr;  // blk.i.attn_norm.weight [hidden]
    const MFTensorView *ffn_norm = nullptr;   // blk.i.ffn_norm.weight  [hidden]

    // MLA
    const MFTensorView *attn_q = nullptr;         // [n_heads*qk_head_dim, hidden]      (out=3072, in=hidden)
    const MFTensorView *attn_kv_a_mqa = nullptr;  // [kv_lora+qk_rope, hidden]          (out=576,  in=hidden)
    const MFTensorView *attn_kv_a_norm = nullptr; // [kv_lora]
    const MFTensorView *attn_kv_b = nullptr;      // [n_heads*(qk_nope+v_head), kv_lora] (out=4096, in=512)
    const MFTensorView *attn_output = nullptr;    // [hidden, n_heads*v_head]           (out=hidden, in=2048)

    bool is_moe = false;

    // dense 层（layer < first_k_dense）
    const MFTensorView *ffn_gate = nullptr; // [dense_ffn, hidden]
    const MFTensorView *ffn_up = nullptr;   // [dense_ffn, hidden]
    const MFTensorView *ffn_down = nullptr; // [hidden, dense_ffn]

    // MoE 层
    const MFTensorView *ffn_gate_inp = nullptr;    // router [n_experts, hidden] F32
    const MFTensorView *ffn_gate_exps = nullptr;   // [n_experts, expert_ffn, hidden]（逐 expert 切片得 [expert_ffn, hidden]）
    const MFTensorView *ffn_up_exps = nullptr;     // [n_experts, expert_ffn, hidden]
    const MFTensorView *ffn_down_exps = nullptr;   // [n_experts, hidden, expert_ffn]
    const MFTensorView *ffn_gate_shexp = nullptr;  // shared [shared_ffn, hidden]
    const MFTensorView *ffn_up_shexp = nullptr;    // [shared_ffn, hidden]
    const MFTensorView *ffn_down_shexp = nullptr;  // [hidden, shared_ffn]
};

class DeepseekWeights {
public:
    DeepseekWeights(const MF &mf, const DeepseekConfig &config);

    // 顶层
    const MFTensorView *token_embd = nullptr;  // [vocab, hidden]
    const MFTensorView *output_norm = nullptr; // [hidden]
    const MFTensorView *output = nullptr;      // lm_head [vocab, hidden]（非 tie）

    std::vector<DeepseekLayerWeights> layers;

private:
    // 外部持有的模型文件；DeepseekWeights 不负责打开/关闭模型文件。
    const MF &mf_;

    // 校验 DeepSeek 推理路径需要的 tensor 是否齐全，且关键 shape 与 config 一致。
    void validate(const DeepseekConfig &config) const;
};

#endif // LOCAL_LLM_DEEPSEEKWEIGHTS_H
