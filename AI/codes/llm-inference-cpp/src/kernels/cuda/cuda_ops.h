#pragma once

#include "../../core/config.h"
#include "cuda_weight_cache.h"
#include "../../model/weights.h"
#include "../../safetensors/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_inference {

// 获取复用的 CUDA hidden buffer，slot 用于 current/next 双缓冲。
void * cuda_token_hidden_buffer(int slot, int hidden_size);

// 获取复用的 CUDA token id buffer；decode 存 generated ids，prefill 存 prompt ids。
void * cuda_token_id_buffer(int count);

// 在设备端读取 token id 并执行 embedding lookup。
bool cuda_embedding_lookup_device_token(const WeightData & emb, const void * device_token_id, void * device_out);

// 对设备端 hidden 做 final RMSNorm 和 logits argmax，结果保留在设备端 token buffer。
bool cuda_final_norm_argmax_to_device(const WeightData & norm_w, const WeightData & emb, const void * device_hidden, int hidden_size, float eps, bool one_plus, void * device_token_out);

// 将设备端生成 token ids 拷贝回主机 vector。
bool cuda_copy_generated_tokens_to_host(const void * device_tokens, int count, std::vector<int> & out);

// 同步当前 CUDA device，失败时返回 false。
bool cuda_synchronize_device();

// 将 float device buffer 转为 BF16/F16 device buffer。
void cuda_float_to_lowp(const float * input, uint16_t * output, int n, cudaDataType_t type);

// 使用 cuBLAS 执行单向量权重乘法，输出 float device buffer。
void cuda_weight_matvec_to_device(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    float * device_y);

// Qwen MLP 的 CUDA backend 执行器，输入为 BF16/F16 device hidden，输出为 float device hidden。
bool cuda_mlp_from_device_bf16_to_device(
    const WeightData & gate_w,
    const WeightData & up_w,
    const WeightData & down_w,
    const uint16_t * device_x,
    float * device_out);

// CUDA 批量 prefill 路径，在 device 上处理完整 prompt，返回最后一个 token 的 device hidden。
const void * cuda_prefill_batch(
    const ModelConfig & config,
    const ModelParams & params,
    const std::vector<int> & prompt_ids,
    std::vector<void *> & linear_states,
    std::vector<void *> & full_states,
    const std::vector<int> & full_max_seq_lens,
    int & seq_len);

// 释放 linear attention 的 CUDA cache/state。
void cuda_free_linear_attention_state(void * state);

// 释放 full attention 的 CUDA KV cache/state。
void cuda_free_full_attention_state(void * state);

} // namespace llm_inference
