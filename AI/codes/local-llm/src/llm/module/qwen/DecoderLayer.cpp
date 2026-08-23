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
#include "tensor/GPUTensor.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/TensorTool.h"

DecoderLayer::DecoderLayer(const LayerWeights &weights, const TextConfig &config)
    : text_config_(config),
      is_full_(weights.type == "full_attention"),
      s_input_norm_weight_(weights.s_input_layernorm),
      s_post_norm_weight_(weights.s_post_attention_layernorm),
      mlp_weights_(weights.mlp) {
    if (is_full_) {
        attn_ = std::make_unique<FullAttention>(weights.full, config);
    } else {
        attn_ = std::make_unique<LinearAttention>(weights.lin, config);
    }
}

void DecoderLayer::prefill(QwenSession &session, const GPUTensor &g_hidden_f32) {
    const size_t input_size = static_cast<size_t>(g_hidden_f32.rows());
    CudaScratch &scratch = session.scratch;
    const int hidden_size = text_config_.hidden_size;
    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};

    GPUTensor g_token_hidden_a_f32 = GPUTensor(scratch, scratch_key::kTokenHiddenA, act_shape, DType::F32);
    GPUTensor g_mixer_f32 = GPUTensor(scratch, scratch_key::kMixer, act_shape, DType::F32);

    // h = x + attn( input_norm(x) )
    RMSNorm::forward(s_input_norm_weight_, g_hidden_f32, g_token_hidden_a_f32,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->prefill(
            session, g_token_hidden_a_f32, g_mixer_f32);
    } else {
        static_cast<LinearAttention *>(attn_.get())->prefill(
            session, g_token_hidden_a_f32, g_mixer_f32);
    }
    TensorTool::add(g_hidden_f32, g_mixer_f32, g_hidden_f32);

    // y = h + mlp( post_norm(h) )
    GPUTensor g_token_hidden_b_f32 = GPUTensor(scratch, scratch_key::kTokenHiddenB, act_shape, DType::F32);
    GPUTensor g_mlp_out_f32 = GPUTensor(scratch, scratch_key::kMlpOut, act_shape, DType::F32);
    RMSNorm::forward(s_post_norm_weight_, g_hidden_f32, g_token_hidden_b_f32,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(mlp_weights_, session, g_token_hidden_b_f32, g_mlp_out_f32);
    TensorTool::add(g_hidden_f32, g_mlp_out_f32, g_hidden_f32);

    // 不做全设备同步：所有 kernel 在同一默认流上按序执行，层间依赖由流内顺序保证；
    // CPU 只在前向末尾 lm_head 的同步 D2H 拷贝处等待一次。此处仅非阻塞检查 launch 错误。
    check_cuda(cudaGetLastError(), "DecoderLayer prefill kernel launch 失败");
}

void DecoderLayer::decode(QwenSession &session, const GPUTensor &g_hidden_f32, int pos) {
    CudaScratch &scratch = session.scratch;
    const int hidden_size = text_config_.hidden_size;
    const std::vector<int64_t> act_shape = {1, static_cast<int64_t>(hidden_size)};

    GPUTensor g_token_hidden_a_f32 = GPUTensor(scratch, scratch_key::kTokenHiddenA, act_shape, DType::F32);
    GPUTensor g_mixer_f32 = GPUTensor(scratch, scratch_key::kMixer, act_shape, DType::F32);

    RMSNorm::forward(s_input_norm_weight_, g_hidden_f32, g_token_hidden_a_f32,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->decode(
            session, g_token_hidden_a_f32, g_mixer_f32, pos);
    } else {
        static_cast<LinearAttention *>(attn_.get())->decode(
            session, g_token_hidden_a_f32, g_mixer_f32);
    }
    TensorTool::add(g_hidden_f32, g_mixer_f32, g_hidden_f32);

    GPUTensor g_token_hidden_b_f32 = GPUTensor(scratch, scratch_key::kTokenHiddenB, act_shape, DType::F32);
    GPUTensor g_mlp_out_f32 = GPUTensor(scratch, scratch_key::kMlpOut, act_shape, DType::F32);
    RMSNorm::forward(s_post_norm_weight_, g_hidden_f32, g_token_hidden_b_f32,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(mlp_weights_, session, g_token_hidden_b_f32, g_mlp_out_f32);
    TensorTool::add(g_hidden_f32, g_mlp_out_f32, g_hidden_f32);

    // 同 prefill：不做全设备同步，依赖同流顺序 + lm_head 末尾同步 D2H 作 barrier。
    check_cuda(cudaGetLastError(), "DecoderLayer decode kernel launch 失败");
}
