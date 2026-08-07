#include "QwenPrefillRunner.h"

#include "../core/cuda_kernels.h"
#include "../kernels/cuda/cache/CudaFullAttentionState.h"
#include "../kernels/cuda/cache/CudaLinearAttentionState.h"
#include "../kernels/cuda/cache/CudaWeightCache.h"
#include "../kernels/cuda/cuda_common.h"
#include "../kernels/cuda/cuda_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace llm_inference {

DeviceWeight & QwenPrefillRunner::require_device_weight(CudaWeightCache & cache, const WeightData & weight, const std::string & context) {
    DeviceWeight * device = cache.cached_weight(weight);
    if (!device) {
        throw std::runtime_error(context + " 失败：权重无法缓存到 CUDA，tensor=" + weight.info->name);
    }
    return *device;
}

bool QwenPrefillRunner::forward_mlp(
        const WeightData & gate_w,
        const WeightData & up_w,
        const WeightData & down_w,
        const uint16_t * device_x,
        int tokens,
        float * device_out) {
    const int intermediate_dim = static_cast<int>(gate_w.info->shape[0]);
    const int hidden_dim = static_cast<int>(gate_w.info->shape[1]);
    if (gate_w.info->dtype != "BF16" || up_w.info->dtype != "BF16" || down_w.info->dtype != "BF16") {
        return false;
    }
    auto & cache = cuda_weight_cache();
    const size_t intermediate_float_bytes = static_cast<size_t>(tokens) * intermediate_dim * sizeof(float);
    const size_t intermediate_lowp_bytes = static_cast<size_t>(tokens) * intermediate_dim * sizeof(uint16_t);
    cache.gate_buffer.ensure_bytes(intermediate_float_bytes, "batch mlp gate");
    cache.up_buffer.ensure_bytes(intermediate_float_bytes, "batch mlp up");
    cache.prod_buffer.ensure_bytes(intermediate_float_bytes, "batch mlp prod");
    cache.prod_lowp_buffer.ensure_bytes(intermediate_lowp_bytes, "batch mlp prod lowp");

    DeviceWeight * gate_up_device = cache.cached_concat_weight(gate_w.info->name + "\n" + up_w.info->name, {gate_w, up_w});
    if (!gate_up_device) {
        return false;
    }
    WeightMeta combined_info = *gate_w.info;
    combined_info.name = gate_w.info->name + "+up";
    combined_info.shape[0] = static_cast<int64_t>(intermediate_dim) * 2;
    WeightData combined_ref {&combined_info, nullptr};
    cache.gate_up_buffer.ensure_bytes(static_cast<size_t>(tokens) * intermediate_dim * 2 * sizeof(float), "batch mlp gate up");
    cuda_weight_batch_matvec_to_device(cache, combined_ref, *gate_up_device, device_x, gate_up_device->type, tokens, cache.gate_up_buffer);
    launch_silu_mul_gate_up_batch(cache.gate_up_buffer, cache.prod_buffer, tokens, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul_gate_up_batch 失败");
    DeviceWeight * down_device = cache.cached_weight(down_w);
    if (!down_device) {
        return false;
    }
    cuda_float_to_lowp(cache.prod_buffer, cache.prod_lowp_buffer, tokens * intermediate_dim, down_device->type);
    check_cuda(cudaGetLastError(), "cuda_float_to_lowp batch mlp prod 失败");
    cuda_weight_batch_matvec_to_device(cache, down_w, *down_device, cache.prod_lowp_buffer, down_device->type, tokens, device_out);
    (void) hidden_dim;
    return true;
}

QwenPrefillRunner::QwenPrefillRunner(const ModelConfig & config, const ModelParams & params)
    : config_(config), params_(params) {
}

Tensor QwenPrefillRunner::forward(const std::vector<int> & input_ids, RunState & state) const {
    std::vector<void *> linear_cuda_states(state.linear.size(), nullptr);
    std::vector<void *> full_cuda_states(state.full.size(), nullptr);
    std::vector<int> full_max_seq_lens(state.full.size(), 0);
    for (size_t i = 0; i < state.linear.size(); ++i) {
        linear_cuda_states[i] = state.linear[i].cuda_state;
    }
    for (size_t i = 0; i < state.full.size(); ++i) {
        full_cuda_states[i] = state.full[i].cuda_state;
        full_max_seq_lens[i] = state.full[i].max_seq_len;
    }

    if (input_ids.empty()) {
        throw std::runtime_error("CUDA batch prefill 失败：prompt ids 为空。");
    }
    if (state.seq_len != 0) {
        throw std::runtime_error("CUDA batch prefill 失败：seq_len 必须从 0 开始。");
    }
    const int tokens = static_cast<int>(input_ids.size());
    const int hidden_dim = config_.text.hidden_size;
    auto & cache = cuda_weight_cache();
    float * current = static_cast<float *>(cuda_token_hidden_buffer(0, tokens * hidden_dim));
    float * next = static_cast<float *>(cuda_token_hidden_buffer(1, tokens * hidden_dim));
    int * token_ids = static_cast<int *>(cuda_token_id_buffer(tokens));
    if (!current || !next || !token_ids) {
        throw std::runtime_error("CUDA batch prefill 失败：token hidden 或 token id buffer 分配失败。");
    }
    check_cuda(cudaMemcpy(token_ids, input_ids.data(), static_cast<size_t>(tokens) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy prompt ids 失败");
    const WeightData emb = params_.embed_tokens;
    DeviceWeight * emb_device = cache.cached_weight(emb);
    if (!emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        throw std::runtime_error("CUDA batch prefill 失败：embedding 权重未缓存或 dtype 不是 BF16/F16。");
    }
    launch_embedding_batch_to_float(
        static_cast<const uint16_t *>(emb_device->ptr),
        token_ids,
        current,
        tokens,
        static_cast<int>(emb.info->shape[0]),
        hidden_dim,
        emb_device->type == CUDA_R_16F ? 1 : 0,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_embedding_batch_to_float 失败");

    const size_t hidden_float_bytes = static_cast<size_t>(tokens) * hidden_dim * sizeof(float);
    const size_t hidden_bf16_bytes = static_cast<size_t>(tokens) * hidden_dim * sizeof(uint16_t);
    cache.mixer_buffer.ensure_bytes(hidden_float_bytes, "batch mixer");
    cache.layer_out_buffer.ensure_bytes(hidden_float_bytes, "batch layer out");
    cache.mlp_out_buffer.ensure_bytes(hidden_float_bytes, "batch mlp out");
    cache.norm_lowp_buffer.ensure_bytes(hidden_bf16_bytes, "batch norm bf16");
    cache.post_norm_lowp_buffer.ensure_bytes(hidden_bf16_bytes, "batch post norm bf16");

    for (int layer = 0; layer < config_.text.num_hidden_layers; ++layer) {
        const LayerWeights & layer_weights = params_.layers[layer];
        const WeightData input_norm_w = layer_weights.input_norm;
        DeviceWeight & input_norm_device = require_device_weight(cache, input_norm_w, "CUDA batch prefill input norm layer=" + std::to_string(layer));
        launch_rms_norm_batch_to_bf16(
            current,
            static_cast<const uint16_t *>(input_norm_device.ptr),
            cache.norm_lowp_buffer,
            tokens,
            hidden_dim,
            config_.text.rms_norm_eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_batch_to_bf16 input 失败");

        if (layer_weights.type == "linear_attention") {
            const int key_heads = config_.text.linear_num_key_heads;
            const int value_heads = config_.text.linear_num_value_heads;
            const int k_dim = config_.text.linear_key_head_dim;
            const int v_dim = config_.text.linear_value_head_dim;
            const int key_total = key_heads * k_dim;
            const int value_total = value_heads * v_dim;
            const int conv_dim = key_total * 2 + value_total;
            CudaLinearAttentionState * linear_state =
                CudaLinearAttentionState::ensure(linear_cuda_states[layer], key_heads, value_heads, k_dim, v_dim, config_.text.linear_conv_kernel_dim);
            linear_state->batch_projection.ensure_bytes(static_cast<size_t>(tokens) * conv_dim * sizeof(float), "batch linear projection");
            linear_state->batch_z.ensure_bytes(static_cast<size_t>(tokens) * value_total * sizeof(float), "batch linear z");
            linear_state->batch_b.ensure_bytes(static_cast<size_t>(tokens) * value_heads * sizeof(float), "batch linear b");
            linear_state->batch_a.ensure_bytes(static_cast<size_t>(tokens) * value_heads * sizeof(float), "batch linear a");
            linear_state->batch_conv_out.ensure_bytes(static_cast<size_t>(tokens) * conv_dim * sizeof(float), "batch linear conv out");
            linear_state->batch_gated.ensure_bytes(static_cast<size_t>(tokens) * value_total * sizeof(float), "batch linear gated");
            linear_state->batch_gated_lowp.ensure_bytes(static_cast<size_t>(tokens) * value_total * sizeof(uint16_t), "batch linear gated bf16");

            const WeightData qkv_w = layer_weights.lin.in_proj_qkv;
            const WeightData z_w = layer_weights.lin.in_proj_z;
            const WeightData b_w = layer_weights.lin.in_proj_b;
            const WeightData a_w = layer_weights.lin.in_proj_a;
            const WeightData conv_w = layer_weights.lin.conv1d;
            const WeightData a_log = layer_weights.lin.a_log;
            const WeightData dt_bias = layer_weights.lin.dt_bias;
            const WeightData attn_norm_w = layer_weights.lin.norm;
            const WeightData attn_out_w = layer_weights.lin.out_proj;
            DeviceWeight & qkv_device = require_device_weight(cache, qkv_w, "CUDA batch prefill linear qkv layer=" + std::to_string(layer));
            cuda_weight_batch_matvec_to_device(cache, qkv_w, qkv_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, linear_state->batch_projection);
            DeviceWeight & z_device = require_device_weight(cache, z_w, "CUDA batch prefill linear z layer=" + std::to_string(layer));
            cuda_weight_batch_matvec_to_device(cache, z_w, z_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, linear_state->batch_z);
            DeviceWeight & b_device = require_device_weight(cache, b_w, "CUDA batch prefill linear b layer=" + std::to_string(layer));
            cuda_weight_batch_matvec_to_device(cache, b_w, b_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, linear_state->batch_b);
            DeviceWeight & a_device = require_device_weight(cache, a_w, "CUDA batch prefill linear a layer=" + std::to_string(layer));
            cuda_weight_batch_matvec_to_device(cache, a_w, a_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, linear_state->batch_a);
            DeviceWeight & conv_device = require_device_weight(cache, conv_w, "CUDA batch prefill linear conv layer=" + std::to_string(layer));
            launch_linear_attention_conv_batch(
                linear_state->batch_projection,
                static_cast<const uint16_t *>(conv_device.ptr),
                linear_state->conv_state,
                linear_state->batch_conv_out,
                tokens,
                conv_dim,
                config_.text.linear_conv_kernel_dim,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_linear_attention_conv_batch 失败");
            DeviceWeight & a_log_device = require_device_weight(cache, a_log, "CUDA batch prefill linear A_log layer=" + std::to_string(layer));
            DeviceWeight & dt_bias_device = require_device_weight(cache, dt_bias, "CUDA batch prefill linear dt_bias layer=" + std::to_string(layer));
            DeviceWeight & attn_norm_device = require_device_weight(cache, attn_norm_w, "CUDA batch prefill linear norm layer=" + std::to_string(layer));
            launch_linear_attention_recurrent_batch(
                linear_state->batch_conv_out,
                linear_state->batch_z,
                linear_state->batch_b,
                linear_state->batch_a,
                static_cast<const float *>(a_log_device.ptr),
                static_cast<const uint16_t *>(dt_bias_device.ptr),
                static_cast<const float *>(attn_norm_device.ptr),
                linear_state->recurrent_state,
                linear_state->batch_gated,
                tokens,
                key_heads,
                value_heads,
                k_dim,
                v_dim,
                config_.text.rms_norm_eps,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent_batch 失败");
            DeviceWeight & attn_out_device = require_device_weight(cache, attn_out_w, "CUDA batch prefill linear out layer=" + std::to_string(layer));
            cuda_float_to_lowp(linear_state->batch_gated, linear_state->batch_gated_lowp, tokens * value_total, attn_out_device.type);
            check_cuda(cudaGetLastError(), "cuda_float_to_lowp batch linear gated 失败");
            cuda_weight_batch_matvec_to_device(cache, attn_out_w, attn_out_device, linear_state->batch_gated_lowp, attn_out_device.type, tokens, cache.mixer_buffer);
        } else {
            const int n_heads = config_.text.num_attention_heads;
            const int kv_heads = config_.text.num_key_value_heads;
            const int head_dim = config_.text.head_dim;
            const int q_total = n_heads * head_dim;
            const int kv_total = kv_heads * head_dim;
            CudaFullAttentionState * full_state =
                CudaFullAttentionState::ensure(full_cuda_states[layer], n_heads, kv_heads, head_dim, full_max_seq_lens[layer]);
            full_state->batch_projection.ensure_bytes(static_cast<size_t>(tokens) * q_total * 2 * sizeof(float), "batch full q projection");
            full_state->batch_k.ensure_bytes(static_cast<size_t>(tokens) * kv_total * sizeof(float), "batch full k");
            full_state->batch_v.ensure_bytes(static_cast<size_t>(tokens) * kv_total * sizeof(float), "batch full v");
            full_state->batch_q.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full q");
            full_state->batch_gate.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full gate");
            full_state->batch_attn.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full attn");
            full_state->batch_attn_lowp.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(uint16_t), "batch full attn bf16");

            const WeightData q_w = layer_weights.full.q_proj;
            const WeightData k_w = layer_weights.full.k_proj;
            const WeightData v_w = layer_weights.full.v_proj;
            const WeightData q_norm_w = layer_weights.full.q_norm;
            const WeightData k_norm_w = layer_weights.full.k_norm;
            const WeightData out_w = layer_weights.full.o_proj;
            DeviceWeight & q_device = require_device_weight(cache, q_w, "CUDA batch prefill full q layer=" + std::to_string(layer));
            cuda_weight_batch_matvec_to_device(cache, q_w, q_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, full_state->batch_projection);
            DeviceWeight & k_device = require_device_weight(cache, k_w, "CUDA batch prefill full k layer=" + std::to_string(layer));
            cuda_weight_batch_matvec_to_device(cache, k_w, k_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, full_state->batch_k);
            DeviceWeight & v_device = require_device_weight(cache, v_w, "CUDA batch prefill full v layer=" + std::to_string(layer));
            cuda_weight_batch_matvec_to_device(cache, v_w, v_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, full_state->batch_v);
            DeviceWeight & q_norm_device = require_device_weight(cache, q_norm_w, "CUDA batch prefill full q_norm layer=" + std::to_string(layer));
            launch_full_attention_q_batch(
                full_state->batch_projection,
                static_cast<const uint16_t *>(q_norm_device.ptr),
                full_state->batch_q,
                full_state->batch_gate,
                tokens,
                n_heads,
                head_dim,
                state.seq_len,
                config_.text.rope_parameters.rope_theta,
                config_.text.rope_parameters.partial_rotary_factor,
                config_.text.rms_norm_eps,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_full_attention_q_batch 失败");
            DeviceWeight & k_norm_device = require_device_weight(cache, k_norm_w, "CUDA batch prefill full k_norm layer=" + std::to_string(layer));
            launch_full_attention_kv_batch(
                full_state->batch_k,
                full_state->batch_v,
                static_cast<const uint16_t *>(k_norm_device.ptr),
                full_state->key_cache,
                full_state->value_cache,
                tokens,
                kv_heads,
                head_dim,
                full_max_seq_lens[layer],
                state.seq_len,
                config_.text.rope_parameters.rope_theta,
                config_.text.rope_parameters.partial_rotary_factor,
                config_.text.rms_norm_eps,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_full_attention_kv_batch 失败");
            launch_full_attention_attend_batch(
                full_state->batch_q,
                full_state->batch_gate,
                full_state->key_cache,
                full_state->value_cache,
                full_state->batch_attn,
                tokens,
                n_heads,
                kv_heads,
                head_dim,
                full_max_seq_lens[layer],
                state.seq_len,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_full_attention_attend_batch 失败");
            DeviceWeight & out_device = require_device_weight(cache, out_w, "CUDA batch prefill full out layer=" + std::to_string(layer));
            cuda_float_to_lowp(full_state->batch_attn, full_state->batch_attn_lowp, tokens * q_total, out_device.type);
            check_cuda(cudaGetLastError(), "cuda_float_to_lowp batch full attn 失败");
            cuda_weight_batch_matvec_to_device(cache, out_w, out_device, full_state->batch_attn_lowp, out_device.type, tokens, cache.mixer_buffer);
        }

        const WeightData post_norm_w = layer_weights.post_norm;
        DeviceWeight & post_norm_device = require_device_weight(cache, post_norm_w, "CUDA batch prefill post norm layer=" + std::to_string(layer));
        launch_add_rms_norm_batch_to_bf16(
            current,
            cache.mixer_buffer,
            static_cast<const uint16_t *>(post_norm_device.ptr),
            cache.layer_out_buffer,
            cache.post_norm_lowp_buffer,
            tokens,
            hidden_dim,
            config_.text.rms_norm_eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_add_rms_norm_batch_to_bf16 post 失败");
        if (!forward_mlp(
                layer_weights.mlp.gate,
                layer_weights.mlp.up,
                layer_weights.mlp.down,
                cache.post_norm_lowp_buffer,
                tokens,
                cache.mlp_out_buffer)) {
            throw std::runtime_error("CUDA batch prefill MLP 失败，layer=" + std::to_string(layer));
        }
        launch_add_float_batch(cache.layer_out_buffer, cache.mlp_out_buffer, next, tokens * hidden_dim, nullptr);
        check_cuda(cudaGetLastError(), "launch_add_float_batch mlp residual 失败");
        std::swap(current, next);
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize batch prefill 失败");
    for (void * state_handle : linear_cuda_states) {
        if (auto * linear_state = static_cast<CudaLinearAttentionState *>(state_handle)) {
            linear_state->release_batch_buffers();
        }
    }
    for (void * state_handle : full_cuda_states) {
        if (auto * full_state = static_cast<CudaFullAttentionState *>(state_handle)) {
            full_state->release_batch_buffers();
        }
    }

    for (size_t i = 0; i < state.linear.size(); ++i) {
        state.linear[i].cuda_state = linear_cuda_states[i];
    }
    for (size_t i = 0; i < state.full.size(); ++i) {
        state.full[i].cuda_state = full_cuda_states[i];
    }
    state.seq_len += tokens;
    return {current + static_cast<size_t>(tokens - 1) * hidden_dim, hidden_dim, -1};
}

} // namespace llm_inference
