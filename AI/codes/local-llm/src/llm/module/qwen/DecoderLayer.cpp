//
// Created by zhangyoulun on 9/8/2026.
//

#include "DecoderLayer.h"

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include "FullAttention.h"
#include "LinearAttention.h"
#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/ops/kernel.cuh"

DecoderLayer::DecoderLayer(const LayerWeights &weights, const TextConfig &config)
    : text_config_(config),
      is_full_(weights.type == "full_attention"),
      input_norm_weight_(weights.input_layernorm),
      post_norm_weight_(weights.post_attention_layernorm),
      mlp_(weights.mlp) {
    if (is_full_) {
        attn_ = std::make_unique<FullAttention>(weights.full, config);
    } else {
        attn_ = std::make_unique<LinearAttention>(weights.lin, config);
    }
}

void DecoderLayer::prefill(QwenSession &session, const Tensor &hidden) {
    float *d_hidden = hidden.gpu_f32();
    const size_t input_size = static_cast<size_t>(hidden.rows());
    CudaScratch &scratch = session.scratch;
    const int hidden_size = text_config_.hidden_size;
    const size_t n = input_size * hidden_size;
    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};

    float *d_token_hidden_a = scratch.ensure<float>(scratch_key::kTokenHiddenA, n);
    float *d_mixer = scratch.ensure<float>(scratch_key::kMixer, n);
    Tensor token_hidden_a = Tensor::gpu_activation(d_token_hidden_a, act_shape);
    Tensor mixer = Tensor::gpu_activation(d_mixer, act_shape);

    // h = x + attn( input_norm(x) )
    RMSNorm::forward(input_norm_weight_, hidden, token_hidden_a,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->prefill(
            session, token_hidden_a, mixer);
    } else {
        static_cast<LinearAttention *>(attn_.get())->prefill(
            session, token_hidden_a, mixer);
    }
    launch_add(d_hidden, d_mixer, d_hidden, static_cast<int>(n), nullptr);

    // y = h + mlp( post_norm(h) )
    float *d_token_hidden_b = scratch.ensure<float>(scratch_key::kTokenHiddenB, n);
    float *d_mlp_out = scratch.ensure<float>(scratch_key::kMlpOut, n);
    Tensor token_hidden_b = Tensor::gpu_activation(d_token_hidden_b, act_shape);
    Tensor mlp_out = Tensor::gpu_activation(d_mlp_out, act_shape);
    RMSNorm::forward(post_norm_weight_, hidden, token_hidden_b,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(session, token_hidden_b, mlp_out);
    launch_add(d_hidden, d_mlp_out, d_hidden, static_cast<int>(n), nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer prefill 同步失败");
}

void DecoderLayer::decode(QwenSession &session, const Tensor &hidden, int pos) {
    float *d_hidden = hidden.gpu_f32();
    CudaScratch &scratch = session.scratch;
    const int hidden_size = text_config_.hidden_size;
    const std::vector<int64_t> act_shape = {1, static_cast<int64_t>(hidden_size)};

    float *d_token_hidden_a = scratch.ensure<float>(scratch_key::kTokenHiddenA, hidden_size);
    float *d_mixer = scratch.ensure<float>(scratch_key::kMixer, hidden_size);
    Tensor token_hidden_a = Tensor::gpu_activation(d_token_hidden_a, act_shape);
    Tensor mixer = Tensor::gpu_activation(d_mixer, act_shape);

    RMSNorm::forward(input_norm_weight_, hidden, token_hidden_a,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->decode(
            session, token_hidden_a, mixer, pos);
    } else {
        static_cast<LinearAttention *>(attn_.get())->decode(
            session, token_hidden_a, mixer);
    }
    launch_add(d_hidden, d_mixer, d_hidden, hidden_size, nullptr);

    float *d_token_hidden_b = scratch.ensure<float>(scratch_key::kTokenHiddenB, hidden_size);
    float *d_mlp_out = scratch.ensure<float>(scratch_key::kMlpOut, hidden_size);
    Tensor token_hidden_b = Tensor::gpu_activation(d_token_hidden_b, act_shape);
    Tensor mlp_out = Tensor::gpu_activation(d_mlp_out, act_shape);
    RMSNorm::forward(post_norm_weight_, hidden, token_hidden_b,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(session, token_hidden_b, mlp_out);
    launch_add(d_hidden, d_mlp_out, d_hidden, hidden_size, nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer decode 同步失败");
}
