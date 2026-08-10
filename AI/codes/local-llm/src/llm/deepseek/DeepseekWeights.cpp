//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekWeights.h"

#include <stdexcept>

const TensorView *DeepseekWeights::get(const TensorContainer &tensors, const std::string &name) {
    if (!tensors.has_tensor(name)) {
        throw std::runtime_error("DeepseekWeights: 缺少张量 " + name);
    }
    return &tensors.tensor(name);
}

const TensorView *DeepseekWeights::opt(const TensorContainer &tensors, const std::string &name) {
    return tensors.has_tensor(name) ? &tensors.tensor(name) : nullptr;
}

DeepseekWeights::DeepseekWeights(const TensorContainer &gguf, const DeepseekConfig &config) {
    token_embd = get(gguf, "token_embd.weight");
    output_norm = get(gguf, "output_norm.weight");
    // V2-Lite lm_head 非 tie，独立 output.weight（Q6_K）；若缺失则回退到 token_embd。
    output = gguf.has_tensor("output.weight") ? get(gguf, "output.weight") : token_embd;

    layers.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        DeepseekLayerWeights &lw = layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";

        lw.attn_norm = get(gguf, p + "attn_norm.weight");
        lw.ffn_norm = get(gguf, p + "ffn_norm.weight");

        lw.attn_q = get(gguf, p + "attn_q.weight");
        lw.attn_kv_a_mqa = get(gguf, p + "attn_kv_a_mqa.weight");
        lw.attn_kv_a_norm = get(gguf, p + "attn_kv_a_norm.weight");
        lw.attn_kv_b = get(gguf, p + "attn_kv_b.weight");
        lw.attn_output = get(gguf, p + "attn_output.weight");

        lw.is_moe = (i >= config.first_k_dense);
        if (!lw.is_moe) {
            lw.ffn_gate = get(gguf, p + "ffn_gate.weight");
            lw.ffn_up = get(gguf, p + "ffn_up.weight");
            lw.ffn_down = get(gguf, p + "ffn_down.weight");
        } else {
            lw.ffn_gate_inp = get(gguf, p + "ffn_gate_inp.weight");
            lw.ffn_gate_exps = get(gguf, p + "ffn_gate_exps.weight");
            lw.ffn_up_exps = get(gguf, p + "ffn_up_exps.weight");
            lw.ffn_down_exps = get(gguf, p + "ffn_down_exps.weight");
            lw.ffn_gate_shexp = get(gguf, p + "ffn_gate_shexp.weight");
            lw.ffn_up_shexp = get(gguf, p + "ffn_up_shexp.weight");
            lw.ffn_down_shexp = get(gguf, p + "ffn_down_shexp.weight");
        }
    }
}
