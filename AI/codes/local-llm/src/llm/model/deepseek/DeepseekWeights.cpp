//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekWeights.h"

DeepseekWeights::DeepseekWeights(const MF &mf, const DeepseekConfig &config) {
    token_embd = &mf.get_tensor_view("token_embd.weight");
    output_norm = &mf.get_tensor_view("output_norm.weight");
    // V2-Lite lm_head 非 tie，独立 output.weight（Q6_K）；若缺失则回退到 token_embd。
    output = mf.contain_tensor_view("output.weight") ? &mf.get_tensor_view("output.weight") : token_embd;

    layers.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        DeepseekLayerWeights &lw = layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";

        lw.attn_norm = &mf.get_tensor_view(p + "attn_norm.weight");
        lw.ffn_norm = &mf.get_tensor_view(p + "ffn_norm.weight");

        lw.attn_q = &mf.get_tensor_view(p + "attn_q.weight");
        lw.attn_kv_a_mqa = &mf.get_tensor_view(p + "attn_kv_a_mqa.weight");
        lw.attn_kv_a_norm = &mf.get_tensor_view(p + "attn_kv_a_norm.weight");
        lw.attn_kv_b = &mf.get_tensor_view(p + "attn_kv_b.weight");
        lw.attn_output = &mf.get_tensor_view(p + "attn_output.weight");

        lw.is_moe = (i >= config.first_k_dense);
        if (!lw.is_moe) {
            lw.ffn_gate = &mf.get_tensor_view(p + "ffn_gate.weight");
            lw.ffn_up = &mf.get_tensor_view(p + "ffn_up.weight");
            lw.ffn_down = &mf.get_tensor_view(p + "ffn_down.weight");
        } else {
            lw.ffn_gate_inp = &mf.get_tensor_view(p + "ffn_gate_inp.weight");
            lw.ffn_gate_exps = &mf.get_tensor_view(p + "ffn_gate_exps.weight");
            lw.ffn_up_exps = &mf.get_tensor_view(p + "ffn_up_exps.weight");
            lw.ffn_down_exps = &mf.get_tensor_view(p + "ffn_down_exps.weight");
            lw.ffn_gate_shexp = &mf.get_tensor_view(p + "ffn_gate_shexp.weight");
            lw.ffn_up_shexp = &mf.get_tensor_view(p + "ffn_up_shexp.weight");
            lw.ffn_down_shexp = &mf.get_tensor_view(p + "ffn_down_shexp.weight");
        }
    }
}
