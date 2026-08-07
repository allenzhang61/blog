#include "cuda_ops.h"
#include "cache/CudaFullAttentionState.h"
#include "cache/CudaLinearAttentionState.h"
#include "cuda_common.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../../core/cuda_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace llm_inference {

namespace {


void cuda_float_to_lowp_impl(const float * input, uint16_t * output, int n, cudaDataType_t type) {
    if (type == CUDA_R_16F) {
        launch_float_to_f16(input, output, n, nullptr);
    } else {
        launch_float_to_bf16(input, output, n, nullptr);
    }
}

void cuda_weight_matvec_to_device_impl(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    float * device_y) {
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasGemmEx(
            cache.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_dim,
            1,
            in_dim,
            &alpha,
            device_weight.ptr,
            device_weight.type,
            in_dim,
            device_x,
            x_type,
            in_dim,
            &beta,
            device_y,
            CUDA_R_32F,
            out_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx matvec 失败 " + weight.info->name);
}

void cublas_batch_matvec_to_device(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    int tokens,
    float * device_y) {
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasGemmEx(
            cache.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_dim,
            tokens,
            in_dim,
            &alpha,
            device_weight.ptr,
            device_weight.type,
            in_dim,
            device_x,
            x_type,
            in_dim,
            &beta,
            device_y,
            CUDA_R_32F,
            out_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx batch matvec 失败 " + weight.info->name);
}

DeviceWeight & require_device_weight(const WeightData & weight, const std::string & context) {
    DeviceWeight * device = cuda_weight_cache().cached_weight(weight);
    if (!device) {
        throw std::runtime_error(context + " 失败：权重无法缓存到 CUDA，tensor=" + weight.info->name);
    }
    return *device;
}

bool cuda_mlp_from_device_bf16_to_device_impl(
    const WeightData & gate_w,
    const WeightData & up_w,
    const WeightData & down_w,
    const uint16_t * device_x,
    float * device_out) {
    const int intermediate_dim = static_cast<int>(gate_w.info->shape[0]);
    const int hidden_dim = static_cast<int>(gate_w.info->shape[1]);
    if (gate_w.info->dtype != "BF16" || up_w.info->dtype != "BF16" || down_w.info->dtype != "BF16") {
        return false;
    }
    auto & cache = cuda_weight_cache();
    DeviceWeight * gate_up_device = cache.cached_concat_weight(gate_w.info->name + "\n" + up_w.info->name, {gate_w, up_w});
    DeviceWeight * down_device = cache.cached_weight(down_w);
    if (!gate_up_device || !down_device) {
        return false;
    }
    const size_t intermediate_float_bytes = static_cast<size_t>(intermediate_dim) * sizeof(float);
    const size_t gate_up_float_bytes = static_cast<size_t>(intermediate_dim) * 2 * sizeof(float);
    const size_t intermediate_bf16_bytes = static_cast<size_t>(intermediate_dim) * sizeof(uint16_t);
    cache.gate_up_buffer.ensure_bytes(gate_up_float_bytes, "mlp gate up buffer");
    cache.prod_buffer.ensure_bytes(intermediate_float_bytes, "mlp prod buffer");
    cache.prod_lowp_buffer.ensure_bytes(intermediate_bf16_bytes, "mlp prod bf16 buffer");

    WeightMeta combined_info = *gate_w.info;
    combined_info.name = gate_w.info->name + "+up";
    combined_info.shape[0] = static_cast<int64_t>(intermediate_dim) * 2;
    WeightData combined_ref {&combined_info, nullptr};
    cuda_weight_matvec_to_device_impl(cache, combined_ref, *gate_up_device, device_x, gate_up_device->type, cache.gate_up_buffer);
    launch_silu_mul(cache.gate_up_buffer, cache.gate_up_buffer + intermediate_dim, cache.prod_buffer, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul 失败");
    cuda_float_to_lowp_impl(cache.prod_buffer, cache.prod_lowp_buffer, intermediate_dim, down_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 失败");
    cuda_weight_matvec_to_device_impl(cache, down_w, *down_device, cache.prod_lowp_buffer, down_device->type, device_out);
    (void) hidden_dim;
    return true;
}

bool cuda_mlp_batch_from_device_bf16_to_device(
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
    cublas_batch_matvec_to_device(cache, combined_ref, *gate_up_device, device_x, gate_up_device->type, tokens, cache.gate_up_buffer);
    launch_silu_mul_gate_up_batch(cache.gate_up_buffer, cache.prod_buffer, tokens, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul_gate_up_batch 失败");
    DeviceWeight * down_device = cache.cached_weight(down_w);
    if (!down_device) {
        return false;
    }
    cuda_float_to_lowp_impl(cache.prod_buffer, cache.prod_lowp_buffer, tokens * intermediate_dim, down_device->type);
    check_cuda(cudaGetLastError(), "cuda_float_to_lowp_impl batch mlp prod 失败");
    cublas_batch_matvec_to_device(cache, down_w, *down_device, cache.prod_lowp_buffer, down_device->type, tokens, device_out);
    (void) hidden_dim;
    return true;
}

} // namespace

void cuda_float_to_lowp(const float * input, uint16_t * output, int n, cudaDataType_t type) {
    cuda_float_to_lowp_impl(input, output, n, type);
}

void cuda_weight_matvec_to_device(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    float * device_y) {
    cuda_weight_matvec_to_device_impl(cache, weight, device_weight, device_x, x_type, device_y);
}

bool cuda_mlp_from_device_bf16_to_device(
    const WeightData & gate_w,
    const WeightData & up_w,
    const WeightData & down_w,
    const uint16_t * device_x,
    float * device_out) {
    return cuda_mlp_from_device_bf16_to_device_impl(gate_w, up_w, down_w, device_x, device_out);
}

void * cuda_token_hidden_buffer(int slot, int hidden_size) {
    auto & cache = cuda_weight_cache();
    CudaScratchBuffer<float> & buffer = slot == 0 ? cache.token_hidden_a : cache.token_hidden_b;
    return buffer.ensure_bytes(static_cast<size_t>(hidden_size) * sizeof(float), slot == 0 ? "token hidden a" : "token hidden b");
}

void * cuda_token_id_buffer(int count) {
    auto & cache = cuda_weight_cache();
    cache.token_id_buffer.ensure_bytes(static_cast<size_t>(count) * sizeof(int), "token id buffer");
    return cache.token_id_buffer;
}

bool cuda_embedding_lookup_device_token(const WeightData & emb, const void * device_token_id, void * device_out) {
    if (emb.info->shape.size() != 2 || !device_token_id || !device_out) {
        return false;
    }
    const int vocab = static_cast<int>(emb.info->shape[0]);
    const int hidden = static_cast<int>(emb.info->shape[1]);
    DeviceWeight * emb_device = cuda_weight_cache().cached_weight(emb);
    if (!emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        return false;
    }
    launch_lowp_embedding_id_to_float(
        static_cast<const uint16_t *>(emb_device->ptr),
        static_cast<const int *>(device_token_id),
        static_cast<float *>(device_out),
        vocab,
        hidden,
        emb_device->type == CUDA_R_16F ? 1 : 0,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_lowp_embedding_id_to_float 失败");
    return true;
}

bool cuda_final_norm_argmax_to_device(
    const WeightData & norm_w,
    const WeightData & emb,
    const void * device_hidden,
    int hidden_size,
    float eps,
    bool one_plus,
    void * device_token_out) {
    if (!device_hidden || !device_token_out || norm_w.info->dtype != "BF16" || norm_w.info->shape.size() != 1 || norm_w.info->shape[0] != hidden_size) {
        return false;
    }
    if (emb.info->shape.size() != 2 || emb.info->shape[1] != hidden_size) {
        return false;
    }
    auto & cache = cuda_weight_cache();
    DeviceWeight * emb_device = cache.cached_weight(emb);
    DeviceWeight * norm_device = cache.cached_weight(norm_w);
    if (!norm_device || !emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        return false;
    }
    const int vocab = static_cast<int>(emb.info->shape[0]);
    cache.norm_lowp_buffer.ensure_bytes(static_cast<size_t>(hidden_size) * sizeof(uint16_t), "final norm lowp");
    cache.y_buffer.ensure_bytes(static_cast<size_t>(vocab) * sizeof(float), "final logits");
    if (emb_device->type == CUDA_R_16F) {
        launch_rms_norm_to_f16(
            static_cast<const float *>(device_hidden),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_lowp_buffer,
            hidden_size,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_f16 final device 失败");
        cuda_weight_matvec_to_device_impl(cache, emb, *emb_device, cache.norm_lowp_buffer, CUDA_R_16F, cache.y_buffer);
    } else {
        launch_rms_norm_to_bf16(
            static_cast<const float *>(device_hidden),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_lowp_buffer,
            hidden_size,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 final device 失败");
        cuda_weight_matvec_to_device_impl(cache, emb, *emb_device, cache.norm_lowp_buffer, CUDA_R_16BF, cache.y_buffer);
    }

    const int blocks = (vocab + 255) / 256;
    cache.argmax_block_values.ensure_bytes(static_cast<size_t>(blocks) * sizeof(float), "argmax block values");
    cache.argmax_block_indices.ensure_bytes(static_cast<size_t>(blocks) * sizeof(int), "argmax block indices");
    cache.argmax_best_value.ensure_bytes(sizeof(float), "argmax best value");
    cache.argmax_best_index.ensure_bytes(sizeof(int), "argmax best index");
    launch_argmax_float(
        cache.y_buffer,
        vocab,
        cache.argmax_block_values,
        cache.argmax_block_indices,
        cache.argmax_best_value,
        cache.argmax_best_index,
        blocks,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_argmax_float final device 失败");
    launch_copy_int(cache.argmax_best_index, static_cast<int *>(device_token_out), nullptr);
    check_cuda(cudaGetLastError(), "launch_copy_int final token 失败");
    return true;
}

bool cuda_copy_generated_tokens_to_host(const void * device_tokens, int count, std::vector<int> & out) {
    if (!device_tokens || count < 0) {
        return false;
    }
    out.assign(static_cast<size_t>(count), 0);
    check_cuda(cudaMemcpy(out.data(), device_tokens, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy generated tokens 失败");
    return true;
}

bool cuda_synchronize_device() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize 失败");
    return true;
}

const void * cuda_prefill_batch(
    const ModelConfig & config,
    const ModelParams & params,
    const std::vector<int> & prompt_ids,
    std::vector<void *> & linear_states,
    std::vector<void *> & full_states,
    const std::vector<int> & full_max_seq_lens,
    int & seq_len) {
    if (prompt_ids.empty()) {
        throw std::runtime_error("CUDA batch prefill 失败：prompt ids 为空。");
    }
    if (seq_len != 0) {
        throw std::runtime_error("CUDA batch prefill 失败：seq_len 必须从 0 开始。");
    }
    const int tokens = static_cast<int>(prompt_ids.size());
    const int hidden_dim = config.text.hidden_size;
    auto & cache = cuda_weight_cache();
    float * current = static_cast<float *>(cuda_token_hidden_buffer(0, tokens * hidden_dim));
    float * next = static_cast<float *>(cuda_token_hidden_buffer(1, tokens * hidden_dim));
    int * token_ids = static_cast<int *>(cuda_token_id_buffer(tokens));
    if (!current || !next || !token_ids) {
        throw std::runtime_error("CUDA batch prefill 失败：token hidden 或 token id buffer 分配失败。");
    }
    check_cuda(cudaMemcpy(token_ids, prompt_ids.data(), static_cast<size_t>(tokens) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy prompt ids 失败");
    const WeightData emb = params.embed_tokens;
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

    for (int layer = 0; layer < config.text.num_hidden_layers; ++layer) {
        const LayerWeights & layer_weights = params.layers[layer];
        const WeightData input_norm_w = layer_weights.input_norm;
        DeviceWeight & input_norm_device = require_device_weight(input_norm_w, "CUDA batch prefill input norm layer=" + std::to_string(layer));
        launch_rms_norm_batch_to_bf16(
            current,
            static_cast<const uint16_t *>(input_norm_device.ptr),
            cache.norm_lowp_buffer,
            tokens,
            hidden_dim,
            config.text.rms_norm_eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_batch_to_bf16 input 失败");

        if (layer_weights.type == "linear_attention") {
            const int key_heads = config.text.linear_num_key_heads;
            const int value_heads = config.text.linear_num_value_heads;
            const int k_dim = config.text.linear_key_head_dim;
            const int v_dim = config.text.linear_value_head_dim;
            const int key_total = key_heads * k_dim;
            const int value_total = value_heads * v_dim;
            const int conv_dim = key_total * 2 + value_total;
            CudaLinearAttentionState * state =
                CudaLinearAttentionState::ensure(linear_states[layer], key_heads, value_heads, k_dim, v_dim, config.text.linear_conv_kernel_dim);
            state->batch_projection.ensure_bytes(static_cast<size_t>(tokens) * conv_dim * sizeof(float), "batch linear projection");
            state->batch_z.ensure_bytes(static_cast<size_t>(tokens) * value_total * sizeof(float), "batch linear z");
            state->batch_b.ensure_bytes(static_cast<size_t>(tokens) * value_heads * sizeof(float), "batch linear b");
            state->batch_a.ensure_bytes(static_cast<size_t>(tokens) * value_heads * sizeof(float), "batch linear a");
            state->batch_conv_out.ensure_bytes(static_cast<size_t>(tokens) * conv_dim * sizeof(float), "batch linear conv out");
            state->batch_gated.ensure_bytes(static_cast<size_t>(tokens) * value_total * sizeof(float), "batch linear gated");
            state->batch_gated_lowp.ensure_bytes(static_cast<size_t>(tokens) * value_total * sizeof(uint16_t), "batch linear gated bf16");

            const WeightData qkv_w = layer_weights.lin.in_proj_qkv;
            const WeightData z_w = layer_weights.lin.in_proj_z;
            const WeightData b_w = layer_weights.lin.in_proj_b;
            const WeightData a_w = layer_weights.lin.in_proj_a;
            const WeightData conv_w = layer_weights.lin.conv1d;
            const WeightData a_log = layer_weights.lin.a_log;
            const WeightData dt_bias = layer_weights.lin.dt_bias;
            const WeightData attn_norm_w = layer_weights.lin.norm;
            const WeightData attn_out_w = layer_weights.lin.out_proj;
            DeviceWeight & qkv_device = require_device_weight(qkv_w, "CUDA batch prefill linear qkv layer=" + std::to_string(layer));
            cublas_batch_matvec_to_device(cache, qkv_w, qkv_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, state->batch_projection);
            DeviceWeight & z_device = require_device_weight(z_w, "CUDA batch prefill linear z layer=" + std::to_string(layer));
            cublas_batch_matvec_to_device(cache, z_w, z_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, state->batch_z);
            DeviceWeight & b_device = require_device_weight(b_w, "CUDA batch prefill linear b layer=" + std::to_string(layer));
            cublas_batch_matvec_to_device(cache, b_w, b_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, state->batch_b);
            DeviceWeight & a_device = require_device_weight(a_w, "CUDA batch prefill linear a layer=" + std::to_string(layer));
            cublas_batch_matvec_to_device(cache, a_w, a_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, state->batch_a);
            DeviceWeight & conv_device = require_device_weight(conv_w, "CUDA batch prefill linear conv layer=" + std::to_string(layer));
            launch_linear_attention_conv_batch(
                state->batch_projection,
                static_cast<const uint16_t *>(conv_device.ptr),
                state->conv_state,
                state->batch_conv_out,
                tokens,
                conv_dim,
                config.text.linear_conv_kernel_dim,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_linear_attention_conv_batch 失败");
            DeviceWeight & a_log_device = require_device_weight(a_log, "CUDA batch prefill linear A_log layer=" + std::to_string(layer));
            DeviceWeight & dt_bias_device = require_device_weight(dt_bias, "CUDA batch prefill linear dt_bias layer=" + std::to_string(layer));
            DeviceWeight & attn_norm_device = require_device_weight(attn_norm_w, "CUDA batch prefill linear norm layer=" + std::to_string(layer));
            launch_linear_attention_recurrent_batch(
                state->batch_conv_out,
                state->batch_z,
                state->batch_b,
                state->batch_a,
                static_cast<const float *>(a_log_device.ptr),
                static_cast<const uint16_t *>(dt_bias_device.ptr),
                static_cast<const float *>(attn_norm_device.ptr),
                state->recurrent_state,
                state->batch_gated,
                tokens,
                key_heads,
                value_heads,
                k_dim,
                v_dim,
                config.text.rms_norm_eps,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent_batch 失败");
            DeviceWeight & attn_out_device = require_device_weight(attn_out_w, "CUDA batch prefill linear out layer=" + std::to_string(layer));
            cuda_float_to_lowp_impl(state->batch_gated, state->batch_gated_lowp, tokens * value_total, attn_out_device.type);
            check_cuda(cudaGetLastError(), "cuda_float_to_lowp_impl batch linear gated 失败");
            cublas_batch_matvec_to_device(cache, attn_out_w, attn_out_device, state->batch_gated_lowp, attn_out_device.type, tokens, cache.mixer_buffer);
        } else {
            const int n_heads = config.text.num_attention_heads;
            const int kv_heads = config.text.num_key_value_heads;
            const int head_dim = config.text.head_dim;
            const int q_total = n_heads * head_dim;
            const int kv_total = kv_heads * head_dim;
            CudaFullAttentionState * state =
                CudaFullAttentionState::ensure(full_states[layer], n_heads, kv_heads, head_dim, full_max_seq_lens[layer]);
            state->batch_projection.ensure_bytes(static_cast<size_t>(tokens) * q_total * 2 * sizeof(float), "batch full q projection");
            state->batch_k.ensure_bytes(static_cast<size_t>(tokens) * kv_total * sizeof(float), "batch full k");
            state->batch_v.ensure_bytes(static_cast<size_t>(tokens) * kv_total * sizeof(float), "batch full v");
            state->batch_q.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full q");
            state->batch_gate.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full gate");
            state->batch_attn.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full attn");
            state->batch_attn_lowp.ensure_bytes(static_cast<size_t>(tokens) * q_total * sizeof(uint16_t), "batch full attn bf16");

            const WeightData q_w = layer_weights.full.q_proj;
            const WeightData k_w = layer_weights.full.k_proj;
            const WeightData v_w = layer_weights.full.v_proj;
            const WeightData q_norm_w = layer_weights.full.q_norm;
            const WeightData k_norm_w = layer_weights.full.k_norm;
            const WeightData out_w = layer_weights.full.o_proj;
            DeviceWeight & q_device = require_device_weight(q_w, "CUDA batch prefill full q layer=" + std::to_string(layer));
            cublas_batch_matvec_to_device(cache, q_w, q_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, state->batch_projection);
            DeviceWeight & k_device = require_device_weight(k_w, "CUDA batch prefill full k layer=" + std::to_string(layer));
            cublas_batch_matvec_to_device(cache, k_w, k_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, state->batch_k);
            DeviceWeight & v_device = require_device_weight(v_w, "CUDA batch prefill full v layer=" + std::to_string(layer));
            cublas_batch_matvec_to_device(cache, v_w, v_device, cache.norm_lowp_buffer, CUDA_R_16BF, tokens, state->batch_v);
            DeviceWeight & q_norm_device = require_device_weight(q_norm_w, "CUDA batch prefill full q_norm layer=" + std::to_string(layer));
            launch_full_attention_q_batch(
                state->batch_projection,
                static_cast<const uint16_t *>(q_norm_device.ptr),
                state->batch_q,
                state->batch_gate,
                tokens,
                n_heads,
                head_dim,
                seq_len,
                config.text.rope_parameters.rope_theta,
                config.text.rope_parameters.partial_rotary_factor,
                config.text.rms_norm_eps,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_full_attention_q_batch 失败");
            DeviceWeight & k_norm_device = require_device_weight(k_norm_w, "CUDA batch prefill full k_norm layer=" + std::to_string(layer));
            launch_full_attention_kv_batch(
                state->batch_k,
                state->batch_v,
                static_cast<const uint16_t *>(k_norm_device.ptr),
                state->key_cache,
                state->value_cache,
                tokens,
                kv_heads,
                head_dim,
                full_max_seq_lens[layer],
                seq_len,
                config.text.rope_parameters.rope_theta,
                config.text.rope_parameters.partial_rotary_factor,
                config.text.rms_norm_eps,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_full_attention_kv_batch 失败");
            launch_full_attention_attend_batch(
                state->batch_q,
                state->batch_gate,
                state->key_cache,
                state->value_cache,
                state->batch_attn,
                tokens,
                n_heads,
                kv_heads,
                head_dim,
                full_max_seq_lens[layer],
                seq_len,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_full_attention_attend_batch 失败");
            DeviceWeight & out_device = require_device_weight(out_w, "CUDA batch prefill full out layer=" + std::to_string(layer));
            cuda_float_to_lowp_impl(state->batch_attn, state->batch_attn_lowp, tokens * q_total, out_device.type);
            check_cuda(cudaGetLastError(), "cuda_float_to_lowp_impl batch full attn 失败");
            cublas_batch_matvec_to_device(cache, out_w, out_device, state->batch_attn_lowp, out_device.type, tokens, cache.mixer_buffer);
        }

        const WeightData post_norm_w = layer_weights.post_norm;
        DeviceWeight & post_norm_device = require_device_weight(post_norm_w, "CUDA batch prefill post norm layer=" + std::to_string(layer));
        launch_add_rms_norm_batch_to_bf16(
            current,
            cache.mixer_buffer,
            static_cast<const uint16_t *>(post_norm_device.ptr),
            cache.layer_out_buffer,
            cache.post_norm_lowp_buffer,
            tokens,
            hidden_dim,
            config.text.rms_norm_eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_add_rms_norm_batch_to_bf16 post 失败");
        if (!cuda_mlp_batch_from_device_bf16_to_device(
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
    for (void * state_handle : linear_states) {
        if (auto * state = static_cast<CudaLinearAttentionState *>(state_handle)) {
            state->release_batch_buffers();
        }
    }
    for (void * state_handle : full_states) {
        if (auto * state = static_cast<CudaFullAttentionState *>(state_handle)) {
            state->release_batch_buffers();
        }
    }
    seq_len += tokens;
    return current + static_cast<size_t>(tokens - 1) * hidden_dim;
}

} // namespace llm_inference
