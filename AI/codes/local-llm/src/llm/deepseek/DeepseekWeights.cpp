//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekWeights.h"

DeepseekWeights::DeepseekWeights(const TensorContainer &gguf, const DeepseekConfig &config) {
    auto required = [&](const std::string &name) {
        return &gguf.get(name);
    };

    token_embd = required("token_embd.weight");
    output_norm = required("output_norm.weight");
    // V2-Lite lm_head 非 tie，独立 output.weight（Q6_K）；若缺失则回退到 token_embd。
    output = gguf.contains("output.weight") ? required("output.weight") : token_embd;

    layers.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        DeepseekLayerWeights &lw = layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";

        lw.attn_norm = required(p + "attn_norm.weight");
        lw.ffn_norm = required(p + "ffn_norm.weight");

        lw.attn_q = required(p + "attn_q.weight");
        lw.attn_kv_a_mqa = required(p + "attn_kv_a_mqa.weight");
        lw.attn_kv_a_norm = required(p + "attn_kv_a_norm.weight");
        lw.attn_kv_b = required(p + "attn_kv_b.weight");
        lw.attn_output = required(p + "attn_output.weight");

        lw.is_moe = (i >= config.first_k_dense);
        if (!lw.is_moe) {
            lw.ffn_gate = required(p + "ffn_gate.weight");
            lw.ffn_up = required(p + "ffn_up.weight");
            lw.ffn_down = required(p + "ffn_down.weight");
        } else {
            lw.ffn_gate_inp = required(p + "ffn_gate_inp.weight");
            lw.ffn_gate_exps = required(p + "ffn_gate_exps.weight");
            lw.ffn_up_exps = required(p + "ffn_up_exps.weight");
            lw.ffn_down_exps = required(p + "ffn_down_exps.weight");
            lw.ffn_gate_shexp = required(p + "ffn_gate_shexp.weight");
            lw.ffn_up_shexp = required(p + "ffn_up_shexp.weight");
            lw.ffn_down_shexp = required(p + "ffn_down_shexp.weight");
        }
    }
}
