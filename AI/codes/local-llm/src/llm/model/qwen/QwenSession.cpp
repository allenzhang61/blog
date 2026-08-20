//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenSession.h"

#include "QwenConfig.h"

QwenSession::QwenSession(const QwenConfig &config, const Tensor &inputs,
                         int max_output_tokens) {
    const TextConfig &text_config = config.data.text;
    max_seq_len = static_cast<size_t>(inputs.numel()) + static_cast<size_t>(max_output_tokens);
    outputs.reserve(static_cast<size_t>(max_output_tokens));

    // full attention 维度。
    const size_t kv_total =
        static_cast<size_t>(text_config.num_key_value_heads) * static_cast<size_t>(text_config.head_dim);

    // linear attention 维度。
    const size_t key_total =
        static_cast<size_t>(text_config.linear_num_key_heads) * static_cast<size_t>(text_config.linear_key_head_dim);
    const size_t value_total =
        static_cast<size_t>(text_config.linear_num_value_heads) * static_cast<size_t>(text_config.linear_value_head_dim);
    const size_t conv_dim = key_total * 2 + value_total;
    const size_t kernel = static_cast<size_t>(text_config.linear_conv_kernel_dim);
    const size_t recurrent_elems = static_cast<size_t>(text_config.linear_num_value_heads) *
                                   static_cast<size_t>(text_config.linear_key_head_dim) *
                                   static_cast<size_t>(text_config.linear_value_head_dim);

    for (const std::string &type : text_config.layer_types) {
        if (type == "full_attention") {
            FullAttnKVCache cache;
            const size_t cache_bytes =
                max_seq_len * kv_total * sizeof(float);
            cache.key_cache = CudaWeight(cache_bytes, CUDA_R_32F, false, "full key cache");
            cache.value_cache = CudaWeight(cache_bytes, CUDA_R_32F, false, "full value cache");
            cache.seq_len = 0;
            full_attn_kv_cache.push_back(std::move(cache));
        } else {
            // linear_attention
            LinearAttnRecurrentState state;
            state.conv_state =
                CudaWeight(conv_dim * kernel * sizeof(float), CUDA_R_32F, true, "linear conv state");
            state.recurrent_state = CudaWeight(recurrent_elems * sizeof(float), CUDA_R_32F, true,
                                               "linear recurrent state");
            linear_attn_recurrent_states.push_back(std::move(state));
        }
    }
}

size_t QwenSession::kv_state_bytes() const {
    size_t total = 0;
    for (const FullAttnKVCache &c : full_attn_kv_cache) {
        total += c.key_cache.bytes + c.value_cache.bytes;
    }
    for (const LinearAttnRecurrentState &s : linear_attn_recurrent_states) {
        total += s.conv_state.bytes + s.recurrent_state.bytes;
    }
    return total;
}
