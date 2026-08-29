//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenSession.h"

#include "QwenConfig.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"

#include <algorithm>
#include <utility>

QwenSession::QwenSession(const QwenConfig &config, std::vector<int> h_input_i32, int max_output_tokens) {
    h_input_i32_ = CPUTensor(cpu_scratch, cpu_scratch_key::kInputIds,
                             {static_cast<int64_t>(h_input_i32.size())}, DType::I32);
    std::copy(h_input_i32.begin(), h_input_i32.end(), h_input_i32_.data<int>());

    const TextConfig &text_config = config.data.text;
    max_seq_len_ = static_cast<size_t>(h_input_i32_.numel()) + static_cast<size_t>(max_output_tokens);
    h_output_i32_.reserve(static_cast<size_t>(max_output_tokens));

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
            // KV cache 改 bf16 存储：显存减半、decode 读 KV 带宽减半（与 llama.cpp f16 KV 口径对齐）。
            // attend/kv kernel 已模板化 KvT，读写时按需转 float 计算。
            const size_t cache_bytes =
                max_seq_len_ * kv_total * sizeof(uint16_t);
            const std::vector<int64_t> cache_shape = {
                static_cast<int64_t>(max_seq_len_),
                static_cast<int64_t>(kv_total),
            };
            cache.g_key_cache_f32 = GPUTensor(
                CudaWeight(cache_bytes, CUDA_R_16BF, false, "full key cache"), cache_shape);
            cache.g_value_cache_f32 = GPUTensor(
                CudaWeight(cache_bytes, CUDA_R_16BF, false, "full value cache"), cache_shape);
            cache.seq_len = 0;
            full_attn_kv_cache.push_back(std::move(cache));
        } else {
            // linear_attention
            LinearAttnRecurrentState state;
            state.g_conv_state_f32 = GPUTensor(
                CudaWeight(conv_dim * kernel * sizeof(float), CUDA_R_32F, true, "linear conv state"),
                {static_cast<int64_t>(conv_dim), static_cast<int64_t>(kernel)});
            // 递归状态改 bf16：显存减半，融合 kernel 读写时按需转 float 计算。
            state.g_recurrent_state_f32 = GPUTensor(
                CudaWeight(recurrent_elems * sizeof(uint16_t), CUDA_R_16BF, true, "linear recurrent state"),
                {static_cast<int64_t>(text_config.linear_num_value_heads),
                 static_cast<int64_t>(text_config.linear_key_head_dim),
                 static_cast<int64_t>(text_config.linear_value_head_dim)});
            linear_attn_recurrent_states.push_back(std::move(state));
        }
    }

    // decode 单步 pos 常驻 device：每步在 graph 外把 host pos 拷进来，kernel 从此读取。
    d_pos_ = GPUTensor(
        CudaWeight(sizeof(int), CUDA_R_32I, false, "decode pos device buffer"), {1});
    // decode 单步 token id 常驻 device：embedding 从此读、argmax 往此写，构成 device 闭环。
    d_token_ = GPUTensor(
        CudaWeight(sizeof(int), CUDA_R_32I, false, "decode token device buffer"), {1});
}

size_t QwenSession::kv_state_bytes() const {
    size_t total = 0;
    for (const FullAttnKVCache &c : full_attn_kv_cache) {
        total += c.g_key_cache_f32.nbytes + c.g_value_cache_f32.nbytes;
    }
    for (const LinearAttnRecurrentState &s : linear_attn_recurrent_states) {
        total += s.g_conv_state_f32.nbytes + s.g_recurrent_state_f32.nbytes;
    }
    return total;
}
