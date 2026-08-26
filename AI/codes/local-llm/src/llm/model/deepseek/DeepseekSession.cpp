//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekSession.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeight.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace {

// ggml/HF YARN 校正维度（返回“pair 索引”单位，范围约 [0, dim/2]）。
float yarn_corr_dim(float num_rot, int dim, float base, int orig_ctx) {
    return (dim * std::log(orig_ctx / (num_rot * 2.0f * static_cast<float>(M_PI)))) /
           (2.0f * std::log(base));
}

} // namespace

DeepseekSession::DeepseekSession(const DeepseekConfig &config, std::vector<int> h_input_i32, int max_output_tokens) {
    h_input_i32_ = CPUTensor(cpu_scratch, cpu_scratch_key::kInputIds,
                             {static_cast<int64_t>(h_input_i32.size())}, DType::I32);
    std::copy(h_input_i32.begin(), h_input_i32.end(), h_input_i32_.data<int>());
    max_seq_len_ = static_cast<size_t>(h_input_i32_.numel()) + static_cast<size_t>(max_output_tokens);

    // latent KV cache：每层 [max_seq_len, kv_lora + qk_rope] float。
    const int kv_total = config.kv_lora_rank + config.qk_rope_head_dim;
    kv_caches.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        const size_t bytes = static_cast<size_t>(max_seq_len_) * kv_total * sizeof(float);
        kv_caches[i].g_cache_f32 = GPUTensor(
            CudaWeight(bytes, CUDA_R_32F, false, "deepseek.kv_cache"),
            {static_cast<int64_t>(max_seq_len_), static_cast<int64_t>(kv_total)});
        kv_caches[i].seq_len = 0;
    }

    // ---- YARN inv_freq ----
    // 精确对齐 llama.cpp rope_yarn / ggml_rope_yarn_corr_dims：
    //   theta_extrap_i = base^(-2i/dim)
    //   theta_interp_i = freq_scale * theta_extrap_i        (freq_scale = 1/scale)
    //   ramp_mix_i     = (1 - clamp((i - low)/(high - low))) * ext_factor   (ext_factor=1)
    //   theta_i        = theta_interp*(1 - ramp_mix) + theta_extrap*ramp_mix
    // 其中 low/high 用 pair 索引 i（对应 ggml col/2）。
    const int dim = config.rope_dim;   // 64
    const int half = dim / 2;          // 32
    const float base = config.rope_theta;
    std::vector<float> h_inv_freq_f32(half);
    if (config.use_yarn) {
        const float scale = config.yarn_scaling_factor;      // 40
        const float freq_scale = 1.0f / scale;
        const int orig = config.yarn_original_context;       // 4096
        float low = std::floor(yarn_corr_dim(config.yarn_beta_fast, dim, base, orig));
        float high = std::ceil(yarn_corr_dim(config.yarn_beta_slow, dim, base, orig));
        low = std::max(low, 0.0f);
        high = std::min(high, static_cast<float>(half - 1));
        for (int i = 0; i < half; ++i) {
            const float theta_extrap = 1.0f / std::pow(base, static_cast<float>(2 * i) / dim);
            const float theta_interp = freq_scale * theta_extrap;
            // rope_yarn_ramp: y=(i-low)/max(0.001,high-low); ramp_mix = 1 - clamp(y,0,1)
            const float y = (static_cast<float>(i) - low) / std::max(0.001f, high - low);
            const float ramp_mix = 1.0f - std::min(std::max(y, 0.0f), 1.0f);
            h_inv_freq_f32[i] = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
    } else {
        for (int i = 0; i < half; ++i) {
            h_inv_freq_f32[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / dim);
        }
    }
    // softmax 缩放固定为 1/sqrt(qk_head_dim)。YARN 的 mscale（yarn_log_multiplier）在
    // llama.cpp 中作用于 RoPE 的 cos/sin 幅值，对短上下文近似为恒等；直接把它塞进
    // softmax 会过度放大注意力打分导致退化（实测验证），故这里不做额外缩放。
    attn_softmax_scale = 1.0f / std::sqrt(static_cast<float>(config.qk_head_dim()));
    g_inv_freq_f32 = CPUTensor(h_inv_freq_f32.data(), {half}, DType::F32)
                     .to_gpu(cuda_scratch, scratch_key::kInvFreq, "deepseek.inv_freq h2d");
}

size_t DeepseekSession::kv_state_bytes() const {
    size_t total = 0;
    for (const auto &kv : kv_caches) {
        total += kv.g_cache_f32.nbytes;
    }
    return total;
}
