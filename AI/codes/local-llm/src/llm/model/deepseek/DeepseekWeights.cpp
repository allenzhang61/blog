//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekWeights.h"

void DeepseekWeights::validate(const DeepseekConfig &config) const {
    mf_.validate();
    mf_.validate_tensor_shape("token_embd.weight", {config.vocab_size, config.hidden_size});
    mf_.validate_tensor_shape("output_norm.weight", {config.hidden_size});

    for (int i = 0; i < config.num_layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const std::string &name : {
                 p + "attn_norm.weight",
                 p + "ffn_norm.weight",
                 p + "attn_q.weight",
                 p + "attn_kv_a_mqa.weight",
                 p + "attn_kv_a_norm.weight",
                 p + "attn_kv_b.weight",
                 p + "attn_output.weight",
             }) {
            mf_.get_tensor_view(name);
        }

        if (i < config.first_k_dense) {
            for (const std::string &name : {
                     p + "ffn_gate.weight",
                     p + "ffn_up.weight",
                     p + "ffn_down.weight",
                 }) {
                mf_.get_tensor_view(name);
            }
        } else {
            for (const std::string &name : {
                     p + "ffn_gate_inp.weight",
                     p + "ffn_gate_exps.weight",
                     p + "ffn_up_exps.weight",
                     p + "ffn_down_exps.weight",
                     p + "ffn_gate_shexp.weight",
                     p + "ffn_up_shexp.weight",
                     p + "ffn_down_shexp.weight",
                 }) {
                mf_.get_tensor_view(name);
            }
        }
    }
}

DeepseekWeights::DeepseekWeights(const MF &mf, const DeepseekConfig &config)
    : mf_(mf) {
    validate(config);

    token_embd = &mf_.get_tensor_view("token_embd.weight");
    output_norm = &mf_.get_tensor_view("output_norm.weight");
    // V2-Lite lm_head 非 tie，独立 output.weight（Q6_K）；若缺失则回退到 token_embd。
    output = mf_.contain_tensor_view("output.weight") ? &mf_.get_tensor_view("output.weight") : token_embd;

    layers.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        DeepseekLayerWeights &lw = layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";

        lw.attn_norm = &mf_.get_tensor_view(p + "attn_norm.weight");
        lw.ffn_norm = &mf_.get_tensor_view(p + "ffn_norm.weight");

        lw.attn_q = &mf_.get_tensor_view(p + "attn_q.weight");
        lw.attn_kv_a_mqa = &mf_.get_tensor_view(p + "attn_kv_a_mqa.weight");
        lw.attn_kv_a_norm = &mf_.get_tensor_view(p + "attn_kv_a_norm.weight");
        lw.attn_kv_b = &mf_.get_tensor_view(p + "attn_kv_b.weight");
        lw.attn_output = &mf_.get_tensor_view(p + "attn_output.weight");

        lw.is_moe = (i >= config.first_k_dense);
        if (!lw.is_moe) {
            lw.ffn_gate = &mf_.get_tensor_view(p + "ffn_gate.weight");
            lw.ffn_up = &mf_.get_tensor_view(p + "ffn_up.weight");
            lw.ffn_down = &mf_.get_tensor_view(p + "ffn_down.weight");
        } else {
            lw.ffn_gate_inp = &mf_.get_tensor_view(p + "ffn_gate_inp.weight");
            lw.ffn_gate_exps = &mf_.get_tensor_view(p + "ffn_gate_exps.weight");
            lw.ffn_up_exps = &mf_.get_tensor_view(p + "ffn_up_exps.weight");
            lw.ffn_down_exps = &mf_.get_tensor_view(p + "ffn_down_exps.weight");
            lw.ffn_gate_shexp = &mf_.get_tensor_view(p + "ffn_gate_shexp.weight");
            lw.ffn_up_shexp = &mf_.get_tensor_view(p + "ffn_up_shexp.weight");
            lw.ffn_down_shexp = &mf_.get_tensor_view(p + "ffn_down_shexp.weight");
        }
    }
}
