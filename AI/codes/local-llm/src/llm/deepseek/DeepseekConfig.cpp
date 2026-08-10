//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekConfig.h"

#include "format/gguf/GgufFile.h"

#include <cmath>
#include <cstdio>

DeepseekConfig::DeepseekConfig(const GgufFile &gguf) {
    hidden_size = static_cast<int>(gguf.metadata_i64("deepseek2.embedding_length"));
    num_layers = static_cast<int>(gguf.metadata_i64("deepseek2.block_count"));
    vocab_size = static_cast<int>(gguf.metadata_i64("deepseek2.vocab_size"));
    num_heads = static_cast<int>(gguf.metadata_i64("deepseek2.attention.head_count"));

    kv_lora_rank = static_cast<int>(gguf.metadata_i64("deepseek2.attention.kv_lora_rank"));
    const int key_length = static_cast<int>(gguf.metadata_i64("deepseek2.attention.key_length"));   // 192
    v_head_dim = static_cast<int>(gguf.metadata_i64("deepseek2.attention.value_length"));            // 128
    rope_dim = static_cast<int>(gguf.metadata_i64("deepseek2.rope.dimension_count"));                // 64
    qk_rope_head_dim = rope_dim;
    qk_nope_head_dim = key_length - qk_rope_head_dim;                                                // 128
    rope_theta = gguf.metadata_f32("deepseek2.rope.freq_base");
    rms_norm_eps = gguf.metadata_f32("deepseek2.attention.layer_norm_rms_epsilon");

    // YARN
    if (gguf.has_metadata("deepseek2.rope.scaling.type") &&
        gguf.metadata_str("deepseek2.rope.scaling.type") == "yarn") {
        use_yarn = true;
        yarn_scaling_factor = gguf.metadata_f32("deepseek2.rope.scaling.factor");
        yarn_original_context =
            static_cast<int>(gguf.metadata_i64("deepseek2.rope.scaling.original_context_length"));
        // yarn_log_multiplier = 0.1 * mscale_all_dim -> 用于 attn softmax 的 mscale。
        const float log_mul = gguf.has_metadata("deepseek2.rope.scaling.yarn_log_multiplier")
                                  ? gguf.metadata_f32("deepseek2.rope.scaling.yarn_log_multiplier")
                                  : 0.0f;
        yarn_mscale = log_mul;      // 直接存 log_multiplier（=0.1*mscale_all_dim）
        yarn_mscale_base = log_mul;
    } else {
        use_yarn = false;
    }

    expert_count = static_cast<int>(gguf.metadata_i64("deepseek2.expert_count"));
    expert_used = static_cast<int>(gguf.metadata_i64("deepseek2.expert_used_count"));
    expert_shared = static_cast<int>(gguf.metadata_i64("deepseek2.expert_shared_count"));
    first_k_dense = static_cast<int>(gguf.metadata_i64("deepseek2.leading_dense_block_count"));
    dense_ffn = static_cast<int>(gguf.metadata_i64("deepseek2.feed_forward_length"));
    expert_ffn = static_cast<int>(gguf.metadata_i64("deepseek2.expert_feed_forward_length"));
    if (gguf.has_metadata("deepseek2.expert_weights_scale")) {
        routed_scaling = gguf.metadata_f32("deepseek2.expert_weights_scale");
    }
    norm_topk_prob = false; // V2-Lite

    eos_token_id = static_cast<int>(gguf.metadata_i64("tokenizer.ggml.eos_token_id"));
    bos_token_id = static_cast<int>(gguf.metadata_i64("tokenizer.ggml.bos_token_id"));
}

void DeepseekConfig::DebugDump() const {
    std::printf(
        "[DeepseekConfig] hidden=%d layers=%d vocab=%d heads=%d | MLA kv_lora=%d qk_nope=%d qk_rope=%d "
        "v_head=%d | rope_theta=%.1f eps=%g yarn=%d factor=%.1f orig_ctx=%d log_mul=%g | MoE experts=%d "
        "used=%d shared=%d first_dense=%d dense_ffn=%d expert_ffn=%d routed_scale=%.3f | eos=%d bos=%d\n",
        hidden_size, num_layers, vocab_size, num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
        v_head_dim, rope_theta, rms_norm_eps, static_cast<int>(use_yarn), yarn_scaling_factor,
        yarn_original_context, yarn_mscale, expert_count, expert_used, expert_shared, first_k_dense,
        dense_ffn, expert_ffn, routed_scaling, eos_token_id, bos_token_id);
}
