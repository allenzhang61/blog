#pragma once

#include "cuda_scratch_buffer.h"

#include "../../safetensors/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace llm_inference {

// 单个已上传到 CUDA device 的权重缓冲区。
class DeviceWeight {
public:
    // device 端权重数据指针。
    void * ptr = nullptr;
    // device 端权重缓冲区字节数。
    size_t bytes = 0;
    // cuBLAS 使用的数据类型。
    cudaDataType_t type = CUDA_R_32F;

    DeviceWeight() = default;
    // 释放 device 端权重数据。
    ~DeviceWeight();
    DeviceWeight(const DeviceWeight &) = delete;
    DeviceWeight & operator=(const DeviceWeight &) = delete;
    // 转移 device 指针所有权，避免重复 cudaFree。
    DeviceWeight(DeviceWeight && other) noexcept;
    // 释放当前缓冲区并接管 other 的 device 指针。
    DeviceWeight & operator=(DeviceWeight && other) noexcept;
};

// CUDA 推理过程复用的全局 cache：cuBLAS handle、权重缓存和临时 device buffer。
class CudaWeightCache {
public:
    // 全局复用的 cuBLAS handle。
    cublasHandle_t handle = nullptr;
    // items 中已缓存权重的总字节数。
    size_t bytes = 0;
    // 通用输入向量 staging buffer。
    CudaScratchBuffer<uint8_t> x_buffer;
    // 通用输出向量 / logits buffer。
    CudaScratchBuffer<float> y_buffer;
    // MLP gate projection 临时输出。
    CudaScratchBuffer<float> gate_buffer;
    // MLP up projection 临时输出。
    CudaScratchBuffer<float> up_buffer;
    // MLP SiLU(gate) * up 的 float 临时结果。
    CudaScratchBuffer<float> prod_buffer;
    // MLP product 转成 BF16/F16 后的低精度输入。
    CudaScratchBuffer<uint16_t> prod_lowp_buffer;
    // 通用子层输出 buffer。
    CudaScratchBuffer<float> out_buffer;
    // 当前层 residual 分支缓存。
    CudaScratchBuffer<float> residual_buffer;
    // attention/mixer 子层输出缓存。
    CudaScratchBuffer<float> mixer_buffer;
    // MLP 子层输出缓存。
    CudaScratchBuffer<float> mlp_out_buffer;
    // 完整 transformer layer 输出缓存。
    CudaScratchBuffer<float> layer_out_buffer;
    // token hidden 双缓冲 A，用于 decode 层间传递。
    CudaScratchBuffer<float> token_hidden_a;
    // token hidden 双缓冲 B，用于 decode 层间传递。
    CudaScratchBuffer<float> token_hidden_b;
    // post-attention RMSNorm 后的低精度 hidden。
    CudaScratchBuffer<uint16_t> post_norm_lowp_buffer;
    // RMSNorm 的 float 输入 staging buffer。
    CudaScratchBuffer<float> norm_input_buffer;
    // RMSNorm 输出的 BF16/F16 hidden。
    CudaScratchBuffer<uint16_t> norm_lowp_buffer;
    // 合并 gate/up projection 后的 float 输出。
    CudaScratchBuffer<float> gate_up_buffer;
    // argmax 每个 block 的最大值缓存。
    CudaScratchBuffer<float> argmax_block_values;
    // argmax 每个 block 的最大值索引缓存。
    CudaScratchBuffer<int> argmax_block_indices;
    // argmax 最终最大值的单元素 device buffer。
    CudaScratchBuffer<float> argmax_best_value;
    // argmax 最终 token id 的单元素 device buffer。
    CudaScratchBuffer<int> argmax_best_index;
    // token ids device staging buffer；decode 保存 generated ids，prefill 保存 prompt ids。
    CudaScratchBuffer<int> token_id_buffer;
    // 按 tensor 名称或组合名称索引的 device 权重缓存。
    std::unordered_map<std::string, DeviceWeight> items;

    // 创建 cuBLAS handle。
    CudaWeightCache();
    // 释放权重缓存和 cuBLAS handle；临时 buffer 由 CudaScratchBuffer 自动释放。
    ~CudaWeightCache();
    CudaWeightCache(const CudaWeightCache &) = delete;
    CudaWeightCache & operator=(const CudaWeightCache &) = delete;
};

// linear attention 层在 CUDA 路径上的跨 token 状态和批量临时 buffer。
class CudaLinearAttentionState {
public:
    int key_heads = 0;
    int value_heads = 0;
    int k_dim = 0;
    int v_dim = 0;
    int kernel = 0;
    float * conv_state = nullptr;
    float * recurrent_state = nullptr;
    float * mixed = nullptr;
    float * projection = nullptr;
    float * z = nullptr;
    float * b = nullptr;
    float * a = nullptr;
    float * conv_out = nullptr;
    float * gated = nullptr;
    uint16_t * gated_bf16 = nullptr;
    CudaScratchBuffer<float> batch_projection;
    CudaScratchBuffer<float> batch_conv_out;
    CudaScratchBuffer<float> batch_gated;
    CudaScratchBuffer<uint16_t> batch_gated_lowp;
    CudaScratchBuffer<float> batch_z;
    CudaScratchBuffer<float> batch_b;
    CudaScratchBuffer<float> batch_a;

    // 释放 linear attention 的 recurrent/conv state 和批量临时 buffer。
    ~CudaLinearAttentionState();
};

// full attention 层在 CUDA 路径上的 KV cache 和批量临时 buffer。
class CudaFullAttentionState {
public:
    int n_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int max_seq_len = 0;
    float * q_and_gate = nullptr;
    float * projection = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    float * q = nullptr;
    float * gate = nullptr;
    float * key_cache = nullptr;
    float * value_cache = nullptr;
    float * attn = nullptr;
    uint16_t * attn_bf16 = nullptr;
    CudaScratchBuffer<float> batch_projection;
    CudaScratchBuffer<float> batch_q;
    CudaScratchBuffer<float> batch_gate;
    CudaScratchBuffer<float> batch_attn;
    CudaScratchBuffer<uint16_t> batch_attn_lowp;
    CudaScratchBuffer<float> batch_k;
    CudaScratchBuffer<float> batch_v;

    // 释放 full attention 的 KV cache 和批量临时 buffer。
    ~CudaFullAttentionState();
};

// 返回进程内唯一的 CUDA cache 实例。
CudaWeightCache & cuda_weight_cache();

// 将 safetensors dtype 映射为 CUDA/cuBLAS dtype。
cudaDataType_t cuda_type_for(const WeightData & weight);

// 返回当前支持 dtype 的单元素字节数。
size_t dtype_size_for(const WeightData & weight);

// 将 host float 编码为 IEEE F16 bit pattern。
uint16_t float_to_f16_bits(float value);

// 将 host float 数组转换为 F16 bit pattern 数组。
std::vector<uint16_t> host_float_to_f16(const std::vector<float> & x);

// 将 BF16 bit pattern 扩展为 host float。
float bf16_to_float(uint16_t value);

// 将 host float 数组转换为 BF16 bit pattern 数组。
std::vector<uint16_t> host_float_to_bf16(const std::vector<float> & x);

// 根据目标 CUDA dtype 将 host float 数组转换为 BF16 或 F16 bit pattern。
std::vector<uint16_t> host_float_to_lowp(const std::vector<float> & x, cudaDataType_t type);

// 获取普通 device 权重缓存；首次访问时从 mmap host 权重上传到 GPU。
DeviceWeight * cached_cuda_weight(const WeightData & weight);

// 将多个 BF16 二维权重按行拼接后上传并缓存，用于合并 projection。
DeviceWeight * cached_cuda_concat_weight(const std::string & name, const std::vector<WeightData> & weights);

// 确保 linear attention CUDA state 已按指定形状初始化。
CudaLinearAttentionState * ensure_linear_attention_state(
    void *& state_handle,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel);

// 确保 full attention CUDA KV cache 已按指定形状初始化。
CudaFullAttentionState * ensure_full_attention_state(
    void *& state_handle,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len);

} // namespace llm_inference
