#include "cuda_ops.h"

#include "../cpu/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
#include "../../core/cuda_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace llm_inference {

namespace {

#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS

size_t cuda_weight_cache_limit_bytes() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_WEIGHT_CACHE_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 10.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

bool cuda_convert_2d_bf16_to_f16() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_CONVERT_BF16_TO_F16");
    return env && std::strcmp(env, "1") == 0;
}

bool cuda_custom_bf16_gemv_enabled() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_CUSTOM_BF16_GEMV");
    return env && std::strcmp(env, "1") == 0;
}

bool cuda_f16_logits_enabled() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_F16_LOGITS");
    return env && std::strcmp(env, "1") == 0;
}

bool cuda_f16_main_gemv_enabled() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_F16_MAIN_GEMV");
    return env && std::strcmp(env, "1") == 0;
}

void check_cuda(cudaError_t status, const std::string & what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(what + "：" + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const std::string & what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(what + "，cublasStatus=" + std::to_string(static_cast<int>(status)));
    }
}

struct DeviceWeight {
    void * ptr = nullptr;
    size_t bytes = 0;
    cudaDataType_t type = CUDA_R_32F;
    ~DeviceWeight() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    DeviceWeight() = default;
    DeviceWeight(const DeviceWeight &) = delete;
    DeviceWeight & operator=(const DeviceWeight &) = delete;
    DeviceWeight(DeviceWeight && other) noexcept {
        ptr = other.ptr;
        bytes = other.bytes;
        type = other.type;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    DeviceWeight & operator=(DeviceWeight && other) noexcept {
        if (this != &other) {
            if (ptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            bytes = other.bytes;
            type = other.type;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
};

struct CudaWeightCache {
    cublasHandle_t handle = nullptr;
    size_t bytes = 0;
    void * x_buffer = nullptr;
    size_t x_bytes = 0;
    float * y_buffer = nullptr;
    size_t y_bytes = 0;
    float * gate_buffer = nullptr;
    size_t gate_bytes = 0;
    float * up_buffer = nullptr;
    size_t up_bytes = 0;
    float * prod_buffer = nullptr;
    size_t prod_bytes = 0;
    uint16_t * prod_bf16_buffer = nullptr;
    size_t prod_bf16_bytes = 0;
    float * out_buffer = nullptr;
    size_t out_bytes = 0;
    float * residual_buffer = nullptr;
    size_t residual_bytes = 0;
    float * mixer_buffer = nullptr;
    size_t mixer_bytes = 0;
    float * mlp_out_buffer = nullptr;
    size_t mlp_out_bytes = 0;
    float * layer_out_buffer = nullptr;
    size_t layer_out_bytes = 0;
    float * token_hidden_a = nullptr;
    size_t token_hidden_a_bytes = 0;
    float * token_hidden_b = nullptr;
    size_t token_hidden_b_bytes = 0;
    uint16_t * post_norm_bf16_buffer = nullptr;
    size_t post_norm_bf16_bytes = 0;
    float * norm_input_buffer = nullptr;
    size_t norm_input_bytes = 0;
    uint16_t * norm_bf16_buffer = nullptr;
    size_t norm_bf16_bytes = 0;
    float * gate_up_buffer = nullptr;
    size_t gate_up_bytes = 0;
    float * argmax_block_values = nullptr;
    size_t argmax_block_values_bytes = 0;
    int * argmax_block_indices = nullptr;
    size_t argmax_block_indices_bytes = 0;
    float * argmax_best_value = nullptr;
    int * argmax_best_index = nullptr;
    int * generated_token_buffer = nullptr;
    size_t generated_token_bytes = 0;
    std::unordered_map<std::string, DeviceWeight> items;
    CudaWeightCache() {
        check_cublas(cublasCreate(&handle), "cublasCreate 失败");
    }
    ~CudaWeightCache() {
        if (x_buffer) {
            cudaFree(x_buffer);
        }
        if (y_buffer) {
            cudaFree(y_buffer);
        }
        if (gate_buffer) {
            cudaFree(gate_buffer);
        }
        if (up_buffer) {
            cudaFree(up_buffer);
        }
        if (prod_buffer) {
            cudaFree(prod_buffer);
        }
        if (prod_bf16_buffer) {
            cudaFree(prod_bf16_buffer);
        }
        if (out_buffer) {
            cudaFree(out_buffer);
        }
        if (residual_buffer) {
            cudaFree(residual_buffer);
        }
        if (mixer_buffer) {
            cudaFree(mixer_buffer);
        }
        if (mlp_out_buffer) {
            cudaFree(mlp_out_buffer);
        }
        if (layer_out_buffer) {
            cudaFree(layer_out_buffer);
        }
        if (token_hidden_a) {
            cudaFree(token_hidden_a);
        }
        if (token_hidden_b) {
            cudaFree(token_hidden_b);
        }
        if (post_norm_bf16_buffer) {
            cudaFree(post_norm_bf16_buffer);
        }
        if (norm_input_buffer) {
            cudaFree(norm_input_buffer);
        }
        if (norm_bf16_buffer) {
            cudaFree(norm_bf16_buffer);
        }
        if (gate_up_buffer) {
            cudaFree(gate_up_buffer);
        }
        if (argmax_block_values) {
            cudaFree(argmax_block_values);
        }
        if (argmax_block_indices) {
            cudaFree(argmax_block_indices);
        }
        if (argmax_best_value) {
            cudaFree(argmax_best_value);
        }
        if (argmax_best_index) {
            cudaFree(argmax_best_index);
        }
        if (generated_token_buffer) {
            cudaFree(generated_token_buffer);
        }
        if (handle) {
            cublasDestroy(handle);
        }
    }
};

struct CudaLinearAttentionState {
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
    float * batch_projection = nullptr;
    size_t batch_projection_bytes = 0;
    float * batch_conv_out = nullptr;
    size_t batch_conv_out_bytes = 0;
    float * batch_gated = nullptr;
    size_t batch_gated_bytes = 0;
    uint16_t * batch_gated_bf16 = nullptr;
    size_t batch_gated_bf16_bytes = 0;
    float * batch_z = nullptr;
    size_t batch_z_bytes = 0;
    float * batch_b = nullptr;
    size_t batch_b_bytes = 0;
    float * batch_a = nullptr;
    size_t batch_a_bytes = 0;

    ~CudaLinearAttentionState() {
        if (conv_state) {
            cudaFree(conv_state);
        }
        if (recurrent_state) {
            cudaFree(recurrent_state);
        }
        if (mixed) {
            cudaFree(mixed);
        }
        if (projection) {
            cudaFree(projection);
        }
        if (z) {
            cudaFree(z);
        }
        if (b) {
            cudaFree(b);
        }
        if (a) {
            cudaFree(a);
        }
        if (conv_out) {
            cudaFree(conv_out);
        }
        if (gated) {
            cudaFree(gated);
        }
        if (gated_bf16) {
            cudaFree(gated_bf16);
        }
        if (batch_projection) {
            cudaFree(batch_projection);
        }
        if (batch_conv_out) {
            cudaFree(batch_conv_out);
        }
        if (batch_gated) {
            cudaFree(batch_gated);
        }
        if (batch_gated_bf16) {
            cudaFree(batch_gated_bf16);
        }
        if (batch_z) {
            cudaFree(batch_z);
        }
        if (batch_b) {
            cudaFree(batch_b);
        }
        if (batch_a) {
            cudaFree(batch_a);
        }
    }
};

struct CudaFullAttentionState {
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
    float * batch_projection = nullptr;
    size_t batch_projection_bytes = 0;
    float * batch_q = nullptr;
    size_t batch_q_bytes = 0;
    float * batch_gate = nullptr;
    size_t batch_gate_bytes = 0;
    float * batch_attn = nullptr;
    size_t batch_attn_bytes = 0;
    uint16_t * batch_attn_bf16 = nullptr;
    size_t batch_attn_bf16_bytes = 0;
    float * batch_k = nullptr;
    size_t batch_k_bytes = 0;
    float * batch_v = nullptr;
    size_t batch_v_bytes = 0;

    ~CudaFullAttentionState() {
        if (q_and_gate) {
            cudaFree(q_and_gate);
        }
        if (projection) {
            cudaFree(projection);
        }
        if (k) {
            cudaFree(k);
        }
        if (v) {
            cudaFree(v);
        }
        if (q) {
            cudaFree(q);
        }
        if (gate) {
            cudaFree(gate);
        }
        if (key_cache) {
            cudaFree(key_cache);
        }
        if (value_cache) {
            cudaFree(value_cache);
        }
        if (attn) {
            cudaFree(attn);
        }
        if (attn_bf16) {
            cudaFree(attn_bf16);
        }
        if (batch_projection) {
            cudaFree(batch_projection);
        }
        if (batch_q) {
            cudaFree(batch_q);
        }
        if (batch_gate) {
            cudaFree(batch_gate);
        }
        if (batch_attn) {
            cudaFree(batch_attn);
        }
        if (batch_attn_bf16) {
            cudaFree(batch_attn_bf16);
        }
        if (batch_k) {
            cudaFree(batch_k);
        }
        if (batch_v) {
            cudaFree(batch_v);
        }
    }
};

CudaWeightCache & cuda_weight_cache() {
    static CudaWeightCache cache;
    return cache;
}

void ensure_cuda_buffers(CudaWeightCache & cache, size_t x_bytes, size_t y_bytes) {
    if (cache.x_bytes < x_bytes) {
        if (cache.x_buffer) {
            cudaFree(cache.x_buffer);
        }
        check_cuda(cudaMalloc(&cache.x_buffer, x_bytes), "cudaMalloc x buffer 失败");
        cache.x_bytes = x_bytes;
    }
    if (cache.y_bytes < y_bytes) {
        if (cache.y_buffer) {
            cudaFree(cache.y_buffer);
        }
        check_cuda(cudaMalloc(&cache.y_buffer, y_bytes), "cudaMalloc y buffer 失败");
        cache.y_bytes = y_bytes;
    }
}

void ensure_float_buffer(float *& ptr, size_t & current_bytes, size_t required_bytes, const std::string & name) {
    if (current_bytes >= required_bytes) {
        return;
    }
    if (ptr) {
        cudaFree(ptr);
    }
    check_cuda(cudaMalloc(&ptr, required_bytes), "cudaMalloc " + name + " 失败");
    current_bytes = required_bytes;
}

void ensure_u16_buffer(uint16_t *& ptr, size_t & current_bytes, size_t required_bytes, const std::string & name) {
    if (current_bytes >= required_bytes) {
        return;
    }
    if (ptr) {
        cudaFree(ptr);
    }
    check_cuda(cudaMalloc(&ptr, required_bytes), "cudaMalloc " + name + " 失败");
    current_bytes = required_bytes;
}

void ensure_int_buffer(int *& ptr, size_t & current_bytes, size_t required_bytes, const std::string & name) {
    if (current_bytes >= required_bytes) {
        return;
    }
    if (ptr) {
        cudaFree(ptr);
    }
    check_cuda(cudaMalloc(&ptr, required_bytes), "cudaMalloc " + name + " 失败");
    current_bytes = required_bytes;
}

cudaDataType_t cuda_type_for(const TensorRef & weight) {
    if (weight.info->dtype == "BF16") {
        return CUDA_R_16BF;
    }
    if (weight.info->dtype == "F16") {
        return CUDA_R_16F;
    }
    if (weight.info->dtype == "F32") {
        return CUDA_R_32F;
    }
    throw std::runtime_error("暂不支持 CUDA dtype：" + weight.info->dtype + " tensor=" + weight.info->name);
}

size_t dtype_size_for(const TensorRef & weight) {
    if (weight.info->dtype == "BF16" || weight.info->dtype == "F16") {
        return sizeof(uint16_t);
    }
    if (weight.info->dtype == "F32") {
        return sizeof(float);
    }
    throw std::runtime_error("暂不支持 dtype：" + weight.info->dtype + " tensor=" + weight.info->name);
}

uint16_t float_to_f16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

std::vector<uint16_t> host_float_to_f16(const std::vector<float> & x) {
    std::vector<uint16_t> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = float_to_f16_bits(x[i]);
    }
    return out;
}

std::vector<uint16_t> host_bf16_tensor_to_f16(const TensorRef & weight) {
    size_t elems = 1;
    for (int64_t dim : weight.info->shape) {
        elems *= static_cast<size_t>(dim);
    }
    std::vector<uint16_t> out(elems);
    const auto * src = reinterpret_cast<const uint16_t *>(weight.data);
    for (size_t i = 0; i < elems; ++i) {
        out[i] = float_to_f16_bits(cpu::bf16_to_float(src[i]));
    }
    return out;
}

DeviceWeight * cached_cuda_weight(const TensorRef & weight) {
    auto & cache = cuda_weight_cache();
    auto found = cache.items.find(weight.info->name);
    if (found != cache.items.end()) {
        return &found->second;
    }

    size_t elems = 1;
    for (int64_t dim : weight.info->shape) {
        elems *= static_cast<size_t>(dim);
    }
    const size_t bytes = elems * dtype_size_for(weight);
    const size_t limit = cuda_weight_cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (cache.bytes + bytes > limit) {
        cache.items.clear();
        cache.bytes = 0;
    }

    DeviceWeight device;
    device.bytes = bytes;
    device.type = cuda_type_for(weight);
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc weight 失败 " + weight.info->name);
    if (weight.info->dtype == "BF16" && weight.info->shape.size() == 2 && cuda_convert_2d_bf16_to_f16()) {
        std::vector<uint16_t> f16 = host_bf16_tensor_to_f16(weight);
        device.type = CUDA_R_16F;
        check_cuda(cudaMemcpy(device.ptr, f16.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy f16 weight 失败 " + weight.info->name);
    } else {
        check_cuda(cudaMemcpy(device.ptr, weight.data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy weight 失败 " + weight.info->name);
    }
    auto [it, inserted] = cache.items.emplace(weight.info->name, std::move(device));
    cache.bytes += bytes;
    (void) inserted;
    return &it->second;
}

DeviceWeight * cached_cuda_weight_f16_copy(const TensorRef & weight) {
    if (weight.info->dtype != "BF16" || weight.info->shape.size() != 2) {
        return nullptr;
    }
    auto & cache = cuda_weight_cache();
    const std::string name = weight.info->name + "#f16";
    auto found = cache.items.find(name);
    if (found != cache.items.end()) {
        return &found->second;
    }

    size_t elems = 1;
    for (int64_t dim : weight.info->shape) {
        elems *= static_cast<size_t>(dim);
    }
    const size_t bytes = elems * sizeof(uint16_t);
    const size_t limit = cuda_weight_cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (cache.bytes + bytes > limit) {
        cache.items.clear();
        cache.bytes = 0;
    }

    std::vector<uint16_t> f16 = host_bf16_tensor_to_f16(weight);
    DeviceWeight device;
    device.bytes = bytes;
    device.type = CUDA_R_16F;
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc f16 weight 失败 " + weight.info->name);
    check_cuda(cudaMemcpy(device.ptr, f16.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy f16 weight 失败 " + weight.info->name);
    auto [it, inserted] = cache.items.emplace(name, std::move(device));
    cache.bytes += bytes;
    (void) inserted;
    return &it->second;
}

DeviceWeight * cached_cuda_matvec_weight(const TensorRef & weight) {
    if (cuda_f16_main_gemv_enabled()) {
        if (DeviceWeight * f16 = cached_cuda_weight_f16_copy(weight)) {
            return f16;
        }
    }
    return cached_cuda_weight(weight);
}

DeviceWeight * cached_cuda_concat_weight(const std::string & name, const std::vector<TensorRef> & weights) {
    auto & cache = cuda_weight_cache();
    auto found = cache.items.find(name);
    if (found != cache.items.end()) {
        return &found->second;
    }
    if (weights.empty()) {
        return nullptr;
    }
    const int64_t in_dim = weights[0].info->shape[1];
    int64_t total_rows = 0;
    for (const TensorRef & weight : weights) {
        if (weight.info->dtype != "BF16" || weight.info->shape.size() != 2 || weight.info->shape[1] != in_dim) {
            return nullptr;
        }
        total_rows += weight.info->shape[0];
    }

    const size_t elems = static_cast<size_t>(total_rows) * static_cast<size_t>(in_dim);
    const size_t bytes = elems * sizeof(uint16_t);
    const size_t limit = cuda_weight_cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (cache.bytes + bytes > limit) {
        cache.items.clear();
        cache.bytes = 0;
    }

    const bool convert_to_f16 = cuda_convert_2d_bf16_to_f16();
    std::vector<uint16_t> host;
    host.reserve(elems);
    for (const TensorRef & weight : weights) {
        const size_t weight_elems = static_cast<size_t>(weight.info->shape[0]) * static_cast<size_t>(in_dim);
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data);
        if (convert_to_f16) {
            for (size_t i = 0; i < weight_elems; ++i) {
                host.push_back(float_to_f16_bits(cpu::bf16_to_float(p[i])));
            }
        } else {
            host.insert(host.end(), p, p + weight_elems);
        }
    }

    DeviceWeight device;
    device.bytes = bytes;
    device.type = convert_to_f16 ? CUDA_R_16F : CUDA_R_16BF;
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc concat weight 失败 " + name);
    check_cuda(cudaMemcpy(device.ptr, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy concat weight 失败 " + name);
    auto [it, inserted] = cache.items.emplace(name, std::move(device));
    cache.bytes += bytes;
    (void) inserted;
    return &it->second;
}

DeviceWeight * cached_cuda_matvec_concat_weight(const std::string & name, const std::vector<TensorRef> & weights) {
    if (!cuda_f16_main_gemv_enabled()) {
        return cached_cuda_concat_weight(name, weights);
    }
    auto & cache = cuda_weight_cache();
    const std::string f16_name = name + "#f16";
    auto found = cache.items.find(f16_name);
    if (found != cache.items.end()) {
        return &found->second;
    }
    if (weights.empty()) {
        return nullptr;
    }
    const int64_t in_dim = weights[0].info->shape[1];
    int64_t total_rows = 0;
    for (const TensorRef & weight : weights) {
        if (weight.info->dtype != "BF16" || weight.info->shape.size() != 2 || weight.info->shape[1] != in_dim) {
            return cached_cuda_concat_weight(name, weights);
        }
        total_rows += weight.info->shape[0];
    }

    const size_t elems = static_cast<size_t>(total_rows) * static_cast<size_t>(in_dim);
    const size_t bytes = elems * sizeof(uint16_t);
    const size_t limit = cuda_weight_cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (cache.bytes + bytes > limit) {
        cache.items.clear();
        cache.bytes = 0;
    }

    std::vector<uint16_t> host;
    host.reserve(elems);
    for (const TensorRef & weight : weights) {
        const size_t weight_elems = static_cast<size_t>(weight.info->shape[0]) * static_cast<size_t>(in_dim);
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data);
        for (size_t i = 0; i < weight_elems; ++i) {
            host.push_back(float_to_f16_bits(cpu::bf16_to_float(p[i])));
        }
    }

    DeviceWeight device;
    device.bytes = bytes;
    device.type = CUDA_R_16F;
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc f16 concat weight 失败 " + name);
    check_cuda(cudaMemcpy(device.ptr, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy f16 concat weight 失败 " + name);
    auto [it, inserted] = cache.items.emplace(f16_name, std::move(device));
    cache.bytes += bytes;
    (void) inserted;
    return &it->second;
}

std::vector<uint16_t> host_float_to_bf16(const std::vector<float> & x) {
    std::vector<uint16_t> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &x[i], sizeof(bits));
        out[i] = static_cast<uint16_t>(bits >> 16);
    }
    return out;
}

std::vector<uint16_t> host_float_to_lowp(const std::vector<float> & x, cudaDataType_t type) {
    if (type == CUDA_R_16F) {
        return host_float_to_f16(x);
    }
    return host_float_to_bf16(x);
}

void launch_float_to_lowp(const float * input, uint16_t * output, int n, cudaDataType_t type) {
    if (type == CUDA_R_16F) {
        launch_float_to_f16(input, output, n, nullptr);
    } else {
        launch_float_to_bf16(input, output, n, nullptr);
    }
}

void cublas_matvec_to_device(
    CudaWeightCache & cache,
    const TensorRef & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    float * device_y) {
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    if (cuda_custom_bf16_gemv_enabled() && device_weight.type == CUDA_R_16BF && x_type == CUDA_R_16BF) {
        launch_bf16_matvec(
            static_cast<const uint16_t *>(device_weight.ptr),
            static_cast<const uint16_t *>(device_x),
            device_y,
            out_dim,
            in_dim,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_bf16_matvec 失败 " + weight.info->name);
        return;
    }
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
    const TensorRef & weight,
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

bool cuda_mlp_from_device_bf16(
    const TensorRef & gate_w,
    const TensorRef & up_w,
    const TensorRef & down_w,
    const uint16_t * device_x,
    std::vector<float> & out) {
    const int intermediate_dim = static_cast<int>(gate_w.info->shape[0]);
    const int hidden_dim = static_cast<int>(gate_w.info->shape[1]);
    if (gate_w.info->dtype != "BF16" || up_w.info->dtype != "BF16" || down_w.info->dtype != "BF16") {
        return false;
    }

    DeviceWeight * gate_up_device = cached_cuda_concat_weight(gate_w.info->name + "\n" + up_w.info->name, {gate_w, up_w});
    DeviceWeight * down_device = cached_cuda_weight(down_w);
    if (!gate_up_device || !down_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t intermediate_float_bytes = static_cast<size_t>(intermediate_dim) * sizeof(float);
    const size_t gate_up_float_bytes = static_cast<size_t>(intermediate_dim) * 2 * sizeof(float);
    const size_t intermediate_bf16_bytes = static_cast<size_t>(intermediate_dim) * sizeof(uint16_t);
    const size_t hidden_float_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    ensure_float_buffer(cache.gate_up_buffer, cache.gate_up_bytes, gate_up_float_bytes, "mlp gate up buffer");
    ensure_float_buffer(cache.prod_buffer, cache.prod_bytes, intermediate_float_bytes, "mlp prod buffer");
    ensure_u16_buffer(cache.prod_bf16_buffer, cache.prod_bf16_bytes, intermediate_bf16_bytes, "mlp prod bf16 buffer");
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, hidden_float_bytes, "mlp out buffer");

    TensorInfo combined_info = *gate_w.info;
    combined_info.name = gate_w.info->name + "+up";
    combined_info.shape[0] = static_cast<int64_t>(intermediate_dim) * 2;
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *gate_up_device, device_x, gate_up_device->type, cache.gate_up_buffer);
    launch_silu_mul(cache.gate_up_buffer, cache.gate_up_buffer + intermediate_dim, cache.prod_buffer, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul 失败");
    launch_float_to_lowp(cache.prod_buffer, cache.prod_bf16_buffer, intermediate_dim, down_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 失败");
    cublas_matvec_to_device(cache, down_w, *down_device, cache.prod_bf16_buffer, down_device->type, cache.out_buffer);

    out.assign(static_cast<size_t>(hidden_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, hidden_float_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy mlp out 失败");
    return true;
}

bool cuda_mlp_from_device_bf16_to_device(
    const TensorRef & gate_w,
    const TensorRef & up_w,
    const TensorRef & down_w,
    const uint16_t * device_x,
    float * device_out) {
    const int intermediate_dim = static_cast<int>(gate_w.info->shape[0]);
    const int hidden_dim = static_cast<int>(gate_w.info->shape[1]);
    if (gate_w.info->dtype != "BF16" || up_w.info->dtype != "BF16" || down_w.info->dtype != "BF16") {
        return false;
    }
    DeviceWeight * gate_up_device = cached_cuda_concat_weight(gate_w.info->name + "\n" + up_w.info->name, {gate_w, up_w});
    DeviceWeight * down_device = cached_cuda_weight(down_w);
    if (!gate_up_device || !down_device) {
        return false;
    }
    auto & cache = cuda_weight_cache();
    const size_t intermediate_float_bytes = static_cast<size_t>(intermediate_dim) * sizeof(float);
    const size_t gate_up_float_bytes = static_cast<size_t>(intermediate_dim) * 2 * sizeof(float);
    const size_t intermediate_bf16_bytes = static_cast<size_t>(intermediate_dim) * sizeof(uint16_t);
    ensure_float_buffer(cache.gate_up_buffer, cache.gate_up_bytes, gate_up_float_bytes, "mlp gate up buffer");
    ensure_float_buffer(cache.prod_buffer, cache.prod_bytes, intermediate_float_bytes, "mlp prod buffer");
    ensure_u16_buffer(cache.prod_bf16_buffer, cache.prod_bf16_bytes, intermediate_bf16_bytes, "mlp prod bf16 buffer");

    TensorInfo combined_info = *gate_w.info;
    combined_info.name = gate_w.info->name + "+up";
    combined_info.shape[0] = static_cast<int64_t>(intermediate_dim) * 2;
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *gate_up_device, device_x, gate_up_device->type, cache.gate_up_buffer);
    launch_silu_mul(cache.gate_up_buffer, cache.gate_up_buffer + intermediate_dim, cache.prod_buffer, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul 失败");
    launch_float_to_lowp(cache.prod_buffer, cache.prod_bf16_buffer, intermediate_dim, down_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 失败");
    cublas_matvec_to_device(cache, down_w, *down_device, cache.prod_bf16_buffer, down_device->type, device_out);
    (void) hidden_dim;
    return true;
}

bool cuda_mlp_batch_from_device_bf16_to_device(
    const TensorRef & gate_w,
    const TensorRef & up_w,
    const TensorRef & down_w,
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
    ensure_float_buffer(cache.gate_buffer, cache.gate_bytes, intermediate_float_bytes, "batch mlp gate");
    ensure_float_buffer(cache.up_buffer, cache.up_bytes, intermediate_float_bytes, "batch mlp up");
    ensure_float_buffer(cache.prod_buffer, cache.prod_bytes, intermediate_float_bytes, "batch mlp prod");
    ensure_u16_buffer(cache.prod_bf16_buffer, cache.prod_bf16_bytes, intermediate_lowp_bytes, "batch mlp prod lowp");

    DeviceWeight * gate_up_device = cached_cuda_concat_weight(gate_w.info->name + "\n" + up_w.info->name, {gate_w, up_w});
    if (!gate_up_device) {
        return false;
    }
    TensorInfo combined_info = *gate_w.info;
    combined_info.name = gate_w.info->name + "+up";
    combined_info.shape[0] = static_cast<int64_t>(intermediate_dim) * 2;
    TensorRef combined_ref {&combined_info, nullptr};
    ensure_float_buffer(cache.gate_up_buffer, cache.gate_up_bytes, static_cast<size_t>(tokens) * intermediate_dim * 2 * sizeof(float), "batch mlp gate up");
    cublas_batch_matvec_to_device(cache, combined_ref, *gate_up_device, device_x, gate_up_device->type, tokens, cache.gate_up_buffer);
    launch_silu_mul_gate_up_batch(cache.gate_up_buffer, cache.prod_buffer, tokens, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul_gate_up_batch 失败");
    DeviceWeight * down_device = cached_cuda_matvec_weight(down_w);
    if (!down_device) {
        return false;
    }
    launch_float_to_lowp(cache.prod_buffer, cache.prod_bf16_buffer, tokens * intermediate_dim, down_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_lowp batch mlp prod 失败");
    cublas_batch_matvec_to_device(cache, down_w, *down_device, cache.prod_bf16_buffer, down_device->type, tokens, device_out);
    (void) hidden_dim;
    return true;
}

CudaLinearAttentionState * ensure_linear_attention_state(
    void *& state_handle,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel) {
    auto * state = static_cast<CudaLinearAttentionState *>(state_handle);
    if (state) {
        if (state->key_heads != key_heads ||
            state->value_heads != value_heads ||
            state->k_dim != k_dim ||
            state->v_dim != v_dim ||
            state->kernel != kernel) {
            throw std::runtime_error("CUDA linear attention state 维度变化，无法复用");
        }
        return state;
    }

    state = new CudaLinearAttentionState();
    state->key_heads = key_heads;
    state->value_heads = value_heads;
    state->k_dim = k_dim;
    state->v_dim = v_dim;
    state->kernel = kernel;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    check_cuda(cudaMalloc(&state->conv_state, static_cast<size_t>(conv_dim) * kernel * sizeof(float)), "cudaMalloc linear conv state 失败");
    check_cuda(cudaMalloc(&state->recurrent_state, static_cast<size_t>(value_heads) * k_dim * v_dim * sizeof(float)), "cudaMalloc linear recurrent state 失败");
    check_cuda(cudaMalloc(&state->mixed, static_cast<size_t>(conv_dim) * sizeof(float)), "cudaMalloc linear mixed 失败");
    check_cuda(cudaMalloc(&state->projection, static_cast<size_t>(conv_dim + value_total + value_heads * 2) * sizeof(float)), "cudaMalloc linear projection 失败");
    check_cuda(cudaMalloc(&state->z, static_cast<size_t>(value_total) * sizeof(float)), "cudaMalloc linear z 失败");
    check_cuda(cudaMalloc(&state->b, static_cast<size_t>(value_heads) * sizeof(float)), "cudaMalloc linear b 失败");
    check_cuda(cudaMalloc(&state->a, static_cast<size_t>(value_heads) * sizeof(float)), "cudaMalloc linear a 失败");
    check_cuda(cudaMalloc(&state->conv_out, static_cast<size_t>(conv_dim) * sizeof(float)), "cudaMalloc linear conv out 失败");
    check_cuda(cudaMalloc(&state->gated, static_cast<size_t>(value_total) * sizeof(float)), "cudaMalloc linear gated 失败");
    check_cuda(cudaMalloc(&state->gated_bf16, static_cast<size_t>(value_total) * sizeof(uint16_t)), "cudaMalloc linear gated bf16 失败");
    check_cuda(cudaMemset(state->conv_state, 0, static_cast<size_t>(conv_dim) * kernel * sizeof(float)), "cudaMemset linear conv state 失败");
    check_cuda(cudaMemset(state->recurrent_state, 0, static_cast<size_t>(value_heads) * k_dim * v_dim * sizeof(float)), "cudaMemset linear recurrent state 失败");
    state_handle = state;
    return state;
}

CudaFullAttentionState * ensure_full_attention_state(
    void *& state_handle,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len) {
    auto * state = static_cast<CudaFullAttentionState *>(state_handle);
    if (state) {
        if (state->n_heads != n_heads ||
            state->kv_heads != kv_heads ||
            state->head_dim != head_dim ||
            state->max_seq_len != max_seq_len) {
            throw std::runtime_error("CUDA full attention state 维度变化，无法复用");
        }
        return state;
    }

    state = new CudaFullAttentionState();
    state->n_heads = n_heads;
    state->kv_heads = kv_heads;
    state->head_dim = head_dim;
    state->max_seq_len = max_seq_len;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    check_cuda(cudaMalloc(&state->q_and_gate, static_cast<size_t>(q_total) * 2 * sizeof(float)), "cudaMalloc full q_and_gate 失败");
    check_cuda(cudaMalloc(&state->projection, static_cast<size_t>(q_total * 2 + kv_total * 2) * sizeof(float)), "cudaMalloc full projection 失败");
    check_cuda(cudaMalloc(&state->k, static_cast<size_t>(kv_total) * sizeof(float)), "cudaMalloc full k 失败");
    check_cuda(cudaMalloc(&state->v, static_cast<size_t>(kv_total) * sizeof(float)), "cudaMalloc full v 失败");
    check_cuda(cudaMalloc(&state->q, static_cast<size_t>(q_total) * sizeof(float)), "cudaMalloc full q 失败");
    check_cuda(cudaMalloc(&state->gate, static_cast<size_t>(q_total) * sizeof(float)), "cudaMalloc full gate 失败");
    check_cuda(cudaMalloc(&state->key_cache, static_cast<size_t>(max_seq_len) * kv_total * sizeof(float)), "cudaMalloc full key cache 失败");
    check_cuda(cudaMalloc(&state->value_cache, static_cast<size_t>(max_seq_len) * kv_total * sizeof(float)), "cudaMalloc full value cache 失败");
    check_cuda(cudaMalloc(&state->attn, static_cast<size_t>(q_total) * sizeof(float)), "cudaMalloc full attn 失败");
    check_cuda(cudaMalloc(&state->attn_bf16, static_cast<size_t>(q_total) * sizeof(uint16_t)), "cudaMalloc full attn bf16 失败");
    state_handle = state;
    return state;
}

void release_linear_attention_batch_buffers(CudaLinearAttentionState * state) {
    if (!state) {
        return;
    }
    if (state->batch_projection) {
        cudaFree(state->batch_projection);
        state->batch_projection = nullptr;
        state->batch_projection_bytes = 0;
    }
    if (state->batch_conv_out) {
        cudaFree(state->batch_conv_out);
        state->batch_conv_out = nullptr;
        state->batch_conv_out_bytes = 0;
    }
    if (state->batch_gated) {
        cudaFree(state->batch_gated);
        state->batch_gated = nullptr;
        state->batch_gated_bytes = 0;
    }
    if (state->batch_gated_bf16) {
        cudaFree(state->batch_gated_bf16);
        state->batch_gated_bf16 = nullptr;
        state->batch_gated_bf16_bytes = 0;
    }
    if (state->batch_z) {
        cudaFree(state->batch_z);
        state->batch_z = nullptr;
        state->batch_z_bytes = 0;
    }
    if (state->batch_b) {
        cudaFree(state->batch_b);
        state->batch_b = nullptr;
        state->batch_b_bytes = 0;
    }
    if (state->batch_a) {
        cudaFree(state->batch_a);
        state->batch_a = nullptr;
        state->batch_a_bytes = 0;
    }
}

void release_full_attention_batch_buffers(CudaFullAttentionState * state) {
    if (!state) {
        return;
    }
    if (state->batch_projection) {
        cudaFree(state->batch_projection);
        state->batch_projection = nullptr;
        state->batch_projection_bytes = 0;
    }
    if (state->batch_q) {
        cudaFree(state->batch_q);
        state->batch_q = nullptr;
        state->batch_q_bytes = 0;
    }
    if (state->batch_gate) {
        cudaFree(state->batch_gate);
        state->batch_gate = nullptr;
        state->batch_gate_bytes = 0;
    }
    if (state->batch_attn) {
        cudaFree(state->batch_attn);
        state->batch_attn = nullptr;
        state->batch_attn_bytes = 0;
    }
    if (state->batch_attn_bf16) {
        cudaFree(state->batch_attn_bf16);
        state->batch_attn_bf16 = nullptr;
        state->batch_attn_bf16_bytes = 0;
    }
    if (state->batch_k) {
        cudaFree(state->batch_k);
        state->batch_k = nullptr;
        state->batch_k_bytes = 0;
    }
    if (state->batch_v) {
        cudaFree(state->batch_v);
        state->batch_v = nullptr;
        state->batch_v_bytes = 0;
    }
}

const uint16_t * cuda_rms_norm_input_to_bf16(
    const TensorRef & norm_w,
    const std::vector<float> & x,
    float eps,
    bool one_plus) {
    if (norm_w.info->dtype != "BF16" || norm_w.info->shape.size() != 1 || norm_w.info->shape[0] != static_cast<int64_t>(x.size())) {
        return nullptr;
    }
    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    if (!norm_device) {
        return nullptr;
    }
    auto & cache = cuda_weight_cache();
    const size_t hidden_float_bytes = x.size() * sizeof(float);
    const size_t hidden_bf16_bytes = x.size() * sizeof(uint16_t);
    ensure_float_buffer(cache.norm_input_buffer, cache.norm_input_bytes, hidden_float_bytes, "norm input buffer");
    ensure_u16_buffer(cache.norm_bf16_buffer, cache.norm_bf16_bytes, hidden_bf16_bytes, "norm bf16 buffer");
    check_cuda(cudaMemcpy(cache.norm_input_buffer, x.data(), hidden_float_bytes, cudaMemcpyHostToDevice), "cudaMemcpy norm input 失败");
    launch_rms_norm_to_bf16(
        cache.norm_input_buffer,
        static_cast<const uint16_t *>(norm_device->ptr),
        cache.norm_bf16_buffer,
        static_cast<int>(x.size()),
        eps,
        one_plus,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 失败");
    return cache.norm_bf16_buffer;
}

#endif

} // namespace

#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS

bool cuda_matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y) {
    DeviceWeight * device_weight = cached_cuda_weight(weight);
    if (!device_weight) {
        return false;
    }
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    y.assign(out_dim, 0.0f);

    auto & cache = cuda_weight_cache();
    std::vector<uint16_t> x_lowp;
    cudaDataType_t x_type = CUDA_R_32F;
    size_t x_bytes = static_cast<size_t>(in_dim) * sizeof(float);
    if (device_weight->type == CUDA_R_16BF || device_weight->type == CUDA_R_16F) {
        x_type = device_weight->type;
        x_bytes = static_cast<size_t>(in_dim) * sizeof(uint16_t);
        x_lowp = host_float_to_lowp(x, x_type);
    }

    const size_t y_bytes = static_cast<size_t>(out_dim) * sizeof(float);
    ensure_cuda_buffers(cache, x_bytes, y_bytes);
    check_cuda(
        cudaMemcpy(
            cache.x_buffer,
            (x_type == CUDA_R_16BF || x_type == CUDA_R_16F) ? static_cast<const void *>(x_lowp.data()) : static_cast<const void *>(x.data()),
            x_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy x 失败");

    cublas_matvec_to_device(cache, weight, *device_weight, cache.x_buffer, x_type, cache.y_buffer);
    check_cuda(cudaMemcpy(y.data(), cache.y_buffer, y_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy y 失败");
    return true;
}

#endif

bool cuda_argmax_matvec(const TensorRef & weight, const std::vector<float> & x, int & best_id) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    DeviceWeight * device_weight = cached_cuda_weight(weight);
    if (!device_weight || (device_weight->type != CUDA_R_16BF && device_weight->type != CUDA_R_16F)) {
        return false;
    }
    if (weight.info->shape.size() != 2) {
        return false;
    }
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    if (static_cast<int>(x.size()) != in_dim) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t x_bytes = static_cast<size_t>(in_dim) * sizeof(uint16_t);
    const size_t y_bytes = static_cast<size_t>(out_dim) * sizeof(float);
    ensure_cuda_buffers(cache, x_bytes, y_bytes);
    std::vector<uint16_t> x_lowp = host_float_to_lowp(x, device_weight->type);
    check_cuda(cudaMemcpy(cache.x_buffer, x_lowp.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy argmax x 失败");
    cublas_matvec_to_device(cache, weight, *device_weight, cache.x_buffer, device_weight->type, cache.y_buffer);

    const int blocks = (out_dim + 255) / 256;
    ensure_float_buffer(
        cache.argmax_block_values,
        cache.argmax_block_values_bytes,
        static_cast<size_t>(blocks) * sizeof(float),
        "argmax block values");
    ensure_int_buffer(
        cache.argmax_block_indices,
        cache.argmax_block_indices_bytes,
        static_cast<size_t>(blocks) * sizeof(int),
        "argmax block indices");
    if (!cache.argmax_best_value) {
        check_cuda(cudaMalloc(&cache.argmax_best_value, sizeof(float)), "cudaMalloc argmax best value 失败");
    }
    if (!cache.argmax_best_index) {
        check_cuda(cudaMalloc(&cache.argmax_best_index, sizeof(int)), "cudaMalloc argmax best index 失败");
    }
    launch_argmax_float(
        cache.y_buffer,
        out_dim,
        cache.argmax_block_values,
        cache.argmax_block_indices,
        cache.argmax_best_value,
        cache.argmax_best_index,
        blocks,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_argmax_float 失败");
    check_cuda(cudaMemcpy(&best_id, cache.argmax_best_index, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy argmax best id 失败");
    return true;
#else
    (void) weight;
    (void) x;
    (void) best_id;
    return false;
#endif
}

void * cuda_token_hidden_buffer(int slot, int hidden_size) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    auto & cache = cuda_weight_cache();
    float *& ptr = slot == 0 ? cache.token_hidden_a : cache.token_hidden_b;
    size_t & bytes = slot == 0 ? cache.token_hidden_a_bytes : cache.token_hidden_b_bytes;
    ensure_float_buffer(ptr, bytes, static_cast<size_t>(hidden_size) * sizeof(float), slot == 0 ? "token hidden a" : "token hidden b");
    return ptr;
#else
    (void) slot;
    (void) hidden_size;
    return nullptr;
#endif
}

void * cuda_generated_token_buffer(int count) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    auto & cache = cuda_weight_cache();
    ensure_int_buffer(cache.generated_token_buffer, cache.generated_token_bytes, static_cast<size_t>(count) * sizeof(int), "generated token buffer");
    return cache.generated_token_buffer;
#else
    (void) count;
    return nullptr;
#endif
}

bool cuda_embedding_lookup_device(const TensorRef & emb, int token_id, void * device_out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (emb.info->shape.size() != 2 || !device_out) {
        return false;
    }
    const int vocab = static_cast<int>(emb.info->shape[0]);
    const int hidden = static_cast<int>(emb.info->shape[1]);
    if (token_id < 0 || token_id >= vocab) {
        throw std::runtime_error("token id 越界：" + std::to_string(token_id));
    }
    DeviceWeight * emb_device = cached_cuda_weight(emb);
    if (!emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        return false;
    }
    const auto * row = static_cast<const uint16_t *>(emb_device->ptr) + static_cast<size_t>(token_id) * hidden;
    launch_lowp_row_to_float(row, static_cast<float *>(device_out), hidden, emb_device->type == CUDA_R_16F ? 1 : 0, nullptr);
    check_cuda(cudaGetLastError(), "launch_lowp_row_to_float embedding 失败");
    return true;
#else
    (void) emb;
    (void) token_id;
    (void) device_out;
    return false;
#endif
}

bool cuda_embedding_lookup_device_token(const TensorRef & emb, const void * device_token_id, void * device_out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (emb.info->shape.size() != 2 || !device_token_id || !device_out) {
        return false;
    }
    const int vocab = static_cast<int>(emb.info->shape[0]);
    const int hidden = static_cast<int>(emb.info->shape[1]);
    DeviceWeight * emb_device = cached_cuda_weight(emb);
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
#else
    (void) emb;
    (void) device_token_id;
    (void) device_out;
    return false;
#endif
}

bool cuda_final_norm_argmax_to_device(
    const TensorRef & norm_w,
    const TensorRef & emb,
    const void * device_hidden,
    int hidden_size,
    float eps,
    bool one_plus,
    void * device_token_out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (!device_hidden || !device_token_out || norm_w.info->dtype != "BF16" || norm_w.info->shape.size() != 1 || norm_w.info->shape[0] != hidden_size) {
        return false;
    }
    if (emb.info->shape.size() != 2 || emb.info->shape[1] != hidden_size) {
        return false;
    }
    DeviceWeight * emb_device = cuda_f16_logits_enabled() ? cached_cuda_weight_f16_copy(emb) : cached_cuda_weight(emb);
    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    if (!norm_device || !emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        return false;
    }
    auto & cache = cuda_weight_cache();
    const int vocab = static_cast<int>(emb.info->shape[0]);
    ensure_u16_buffer(cache.norm_bf16_buffer, cache.norm_bf16_bytes, static_cast<size_t>(hidden_size) * sizeof(uint16_t), "final norm lowp");
    ensure_float_buffer(cache.y_buffer, cache.y_bytes, static_cast<size_t>(vocab) * sizeof(float), "final logits");
    if (emb_device->type == CUDA_R_16F) {
        launch_rms_norm_to_f16(
            static_cast<const float *>(device_hidden),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_size,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_f16 final device 失败");
        cublas_matvec_to_device(cache, emb, *emb_device, cache.norm_bf16_buffer, CUDA_R_16F, cache.y_buffer);
    } else {
        launch_rms_norm_to_bf16(
            static_cast<const float *>(device_hidden),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_size,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 final device 失败");
        cublas_matvec_to_device(cache, emb, *emb_device, cache.norm_bf16_buffer, CUDA_R_16BF, cache.y_buffer);
    }

    const int blocks = (vocab + 255) / 256;
    ensure_float_buffer(cache.argmax_block_values, cache.argmax_block_values_bytes, static_cast<size_t>(blocks) * sizeof(float), "argmax block values");
    ensure_int_buffer(cache.argmax_block_indices, cache.argmax_block_indices_bytes, static_cast<size_t>(blocks) * sizeof(int), "argmax block indices");
    if (!cache.argmax_best_value) {
        check_cuda(cudaMalloc(&cache.argmax_best_value, sizeof(float)), "cudaMalloc argmax best value 失败");
    }
    if (!cache.argmax_best_index) {
        check_cuda(cudaMalloc(&cache.argmax_best_index, sizeof(int)), "cudaMalloc argmax best index 失败");
    }
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
#else
    (void) norm_w;
    (void) emb;
    (void) device_hidden;
    (void) hidden_size;
    (void) eps;
    (void) one_plus;
    (void) device_token_out;
    return false;
#endif
}

bool cuda_copy_generated_tokens_to_host(const void * device_tokens, int count, std::vector<int> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (!device_tokens || count < 0) {
        return false;
    }
    out.assign(static_cast<size_t>(count), 0);
    check_cuda(cudaMemcpy(out.data(), device_tokens, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy generated tokens 失败");
    return true;
#else
    (void) device_tokens;
    (void) count;
    (void) out;
    return false;
#endif
}

bool cuda_synchronize_device() {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize 失败");
    return true;
#else
    return false;
#endif
}

bool cuda_final_norm_argmax_device(
    const TensorRef & norm_w,
    const TensorRef & emb,
    const void * device_hidden,
    int hidden_size,
    float eps,
    bool one_plus,
    int & best_id) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (!device_hidden || norm_w.info->dtype != "BF16" || norm_w.info->shape.size() != 1 || norm_w.info->shape[0] != hidden_size) {
        return false;
    }
    if (emb.info->shape.size() != 2 || emb.info->shape[1] != hidden_size) {
        return false;
    }
    DeviceWeight * emb_device = cuda_f16_logits_enabled() ? cached_cuda_weight_f16_copy(emb) : cached_cuda_weight(emb);
    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    if (!norm_device || !emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        return false;
    }
    auto & cache = cuda_weight_cache();
    const int vocab = static_cast<int>(emb.info->shape[0]);
    ensure_u16_buffer(cache.norm_bf16_buffer, cache.norm_bf16_bytes, static_cast<size_t>(hidden_size) * sizeof(uint16_t), "final norm bf16");
    ensure_float_buffer(cache.y_buffer, cache.y_bytes, static_cast<size_t>(vocab) * sizeof(float), "final logits");
    if (emb_device->type == CUDA_R_16F) {
        launch_rms_norm_to_f16(
            static_cast<const float *>(device_hidden),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_size,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_f16 final 失败");
        cublas_matvec_to_device(cache, emb, *emb_device, cache.norm_bf16_buffer, CUDA_R_16F, cache.y_buffer);
    } else {
        launch_rms_norm_to_bf16(
            static_cast<const float *>(device_hidden),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_size,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 final 失败");
        cublas_matvec_to_device(cache, emb, *emb_device, cache.norm_bf16_buffer, CUDA_R_16BF, cache.y_buffer);
    }

    const int blocks = (vocab + 255) / 256;
    ensure_float_buffer(cache.argmax_block_values, cache.argmax_block_values_bytes, static_cast<size_t>(blocks) * sizeof(float), "argmax block values");
    ensure_int_buffer(cache.argmax_block_indices, cache.argmax_block_indices_bytes, static_cast<size_t>(blocks) * sizeof(int), "argmax block indices");
    if (!cache.argmax_best_value) {
        check_cuda(cudaMalloc(&cache.argmax_best_value, sizeof(float)), "cudaMalloc argmax best value 失败");
    }
    if (!cache.argmax_best_index) {
        check_cuda(cudaMalloc(&cache.argmax_best_index, sizeof(int)), "cudaMalloc argmax best index 失败");
    }
    launch_argmax_float(
        cache.y_buffer,
        vocab,
        cache.argmax_block_values,
        cache.argmax_block_indices,
        cache.argmax_best_value,
        cache.argmax_best_index,
        blocks,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_argmax_float final 失败");
    check_cuda(cudaMemcpy(&best_id, cache.argmax_best_index, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy final argmax id 失败");
    return true;
#else
    (void) norm_w;
    (void) emb;
    (void) device_hidden;
    (void) hidden_size;
    (void) eps;
    (void) one_plus;
    (void) best_id;
    return false;
#endif
}

bool cuda_mlp_layer(
    const TensorRef & gate_w,
    const TensorRef & up_w,
    const TensorRef & down_w,
    const std::vector<float> & x,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (gate_w.info->shape.size() != 2 || up_w.info->shape.size() != 2 || down_w.info->shape.size() != 2) {
        return false;
    }
    if (gate_w.info->dtype != "BF16" || up_w.info->dtype != "BF16" || down_w.info->dtype != "BF16") {
        return false;
    }

    const int intermediate_dim = static_cast<int>(gate_w.info->shape[0]);
    const int hidden_dim = static_cast<int>(gate_w.info->shape[1]);
    if (up_w.info->shape[0] != intermediate_dim ||
        up_w.info->shape[1] != hidden_dim ||
        down_w.info->shape[0] != hidden_dim ||
        down_w.info->shape[1] != intermediate_dim ||
        static_cast<int>(x.size()) != hidden_dim) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    DeviceWeight * gate_up_device = cached_cuda_concat_weight(gate_w.info->name + "\n" + up_w.info->name, {gate_w, up_w});
    if (!gate_up_device) {
        return false;
    }
    const size_t x_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_cuda_buffers(cache, x_bytes, 1);

    std::vector<uint16_t> x_lowp = host_float_to_lowp(x, gate_up_device->type);
    check_cuda(cudaMemcpy(cache.x_buffer, x_lowp.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy mlp x 失败");
    return cuda_mlp_from_device_bf16(gate_w, up_w, down_w, static_cast<const uint16_t *>(cache.x_buffer), out);
#else
    (void) gate_w;
    (void) up_w;
    (void) down_w;
    (void) x;
    (void) out;
    return false;
#endif
}

bool cuda_linear_attention_layer(
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & norm_w,
    const TensorRef & out_w,
    const std::vector<float> & mixed,
    const std::vector<float> & z,
    const std::vector<float> & b,
    const std::vector<float> & a,
    void *& state_handle,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    if (conv_w.info->dtype != "BF16" ||
        a_log.info->dtype != "F32" ||
        dt_bias.info->dtype != "BF16" ||
        norm_w.info->dtype != "F32" ||
        out_w.info->dtype != "BF16") {
        return false;
    }
    if (static_cast<int>(mixed.size()) != conv_dim ||
        static_cast<int>(z.size()) != value_total ||
        static_cast<int>(b.size()) != value_heads ||
        static_cast<int>(a.size()) != value_heads) {
        return false;
    }
    if (conv_w.info->shape.size() != 3 ||
        conv_w.info->shape[0] != conv_dim ||
        conv_w.info->shape[2] != kernel ||
        a_log.info->shape.size() != 1 ||
        a_log.info->shape[0] != value_heads ||
        dt_bias.info->shape.size() != 1 ||
        dt_bias.info->shape[0] != value_heads ||
        norm_w.info->shape.size() != 1 ||
        norm_w.info->shape[0] != v_dim ||
        out_w.info->shape.size() != 2 ||
        out_w.info->shape[1] != value_total) {
        return false;
    }

    DeviceWeight * conv_device = cached_cuda_weight(conv_w);
    DeviceWeight * a_log_device = cached_cuda_weight(a_log);
    DeviceWeight * dt_bias_device = cached_cuda_weight(dt_bias);
    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!conv_device || !a_log_device || !dt_bias_device || !norm_device || !out_device) {
        return false;
    }

    CudaLinearAttentionState * state =
        ensure_linear_attention_state(state_handle, key_heads, value_heads, k_dim, v_dim, kernel);
    check_cuda(cudaMemcpy(state->mixed, mixed.data(), static_cast<size_t>(conv_dim) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy linear mixed 失败");
    check_cuda(cudaMemcpy(state->z, z.data(), static_cast<size_t>(value_total) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy linear z 失败");
    check_cuda(cudaMemcpy(state->b, b.data(), static_cast<size_t>(value_heads) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy linear b 失败");
    check_cuda(cudaMemcpy(state->a, a.data(), static_cast<size_t>(value_heads) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy linear a 失败");

    launch_linear_attention_conv(
        state->mixed,
        static_cast<const uint16_t *>(conv_device->ptr),
        state->conv_state,
        state->conv_out,
        conv_dim,
        kernel,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_conv 失败");
    launch_linear_attention_recurrent(
        state->conv_out,
        state->z,
        state->b,
        state->a,
        static_cast<const float *>(a_log_device->ptr),
        static_cast<const uint16_t *>(dt_bias_device->ptr),
        static_cast<const float *>(norm_device->ptr),
        state->recurrent_state,
        state->gated,
        key_heads,
        value_heads,
        k_dim,
        v_dim,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent 失败");
    launch_float_to_bf16(state->gated, state->gated_bf16, value_total, nullptr);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 linear gated 失败");

    auto & cache = cuda_weight_cache();
    const int hidden_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(hidden_dim) * sizeof(float), "linear out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->gated_bf16, CUDA_R_16BF, cache.out_buffer);

    out.assign(static_cast<size_t>(hidden_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, static_cast<size_t>(hidden_dim) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy linear out 失败");
    return true;
#else
    (void) conv_w;
    (void) a_log;
    (void) dt_bias;
    (void) norm_w;
    (void) out_w;
    (void) mixed;
    (void) z;
    (void) b;
    (void) a;
    (void) state_handle;
    (void) key_heads;
    (void) value_heads;
    (void) k_dim;
    (void) v_dim;
    (void) kernel;
    (void) eps;
    (void) out;
    return false;
#endif
}

bool cuda_linear_attention_project_layer(
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state_handle,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    if (in_proj_qkv_w.info->dtype != "BF16" ||
        in_proj_z_w.info->dtype != "BF16" ||
        in_proj_b_w.info->dtype != "BF16" ||
        in_proj_a_w.info->dtype != "BF16" ||
        conv_w.info->dtype != "BF16" ||
        a_log.info->dtype != "F32" ||
        dt_bias.info->dtype != "BF16" ||
        norm_w.info->dtype != "F32" ||
        out_w.info->dtype != "BF16") {
        return false;
    }
    if (static_cast<int>(x.size()) != in_proj_qkv_w.info->shape[1] ||
        in_proj_qkv_w.info->shape[0] != conv_dim ||
        in_proj_z_w.info->shape[0] != value_total ||
        in_proj_b_w.info->shape[0] != value_heads ||
        in_proj_a_w.info->shape[0] != value_heads) {
        return false;
    }

    DeviceWeight * projection_device = cached_cuda_matvec_concat_weight(
        in_proj_qkv_w.info->name + "\n" + in_proj_z_w.info->name + "\n" + in_proj_b_w.info->name + "\n" + in_proj_a_w.info->name,
        {in_proj_qkv_w, in_proj_z_w, in_proj_b_w, in_proj_a_w});
    DeviceWeight * conv_device = cached_cuda_weight(conv_w);
    DeviceWeight * a_log_device = cached_cuda_weight(a_log);
    DeviceWeight * dt_bias_device = cached_cuda_weight(dt_bias);
    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!projection_device || !conv_device || !a_log_device || !dt_bias_device || !norm_device || !out_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const int hidden_dim = static_cast<int>(x.size());
    const size_t x_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_cuda_buffers(cache, x_bytes, 1);
    std::vector<uint16_t> x_lowp = host_float_to_lowp(x, projection_device->type);
    check_cuda(cudaMemcpy(cache.x_buffer, x_lowp.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy linear project x 失败");

    CudaLinearAttentionState * state =
        ensure_linear_attention_state(state_handle, key_heads, value_heads, k_dim, v_dim, kernel);
    TensorInfo combined_info = *in_proj_qkv_w.info;
    combined_info.name = in_proj_qkv_w.info->name + "+z+b+a";
    combined_info.shape[0] = static_cast<int64_t>(conv_dim + value_total + value_heads * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, cache.x_buffer, projection_device->type, state->projection);
    const float * mixed_ptr = state->projection;
    const float * z_ptr = state->projection + conv_dim;
    const float * b_ptr = state->projection + conv_dim + value_total;
    const float * a_ptr = state->projection + conv_dim + value_total + value_heads;

    launch_linear_attention_conv(
        mixed_ptr,
        static_cast<const uint16_t *>(conv_device->ptr),
        state->conv_state,
        state->conv_out,
        conv_dim,
        kernel,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_conv 失败");
    launch_linear_attention_recurrent(
        state->conv_out,
        z_ptr,
        b_ptr,
        a_ptr,
        static_cast<const float *>(a_log_device->ptr),
        static_cast<const uint16_t *>(dt_bias_device->ptr),
        static_cast<const float *>(norm_device->ptr),
        state->recurrent_state,
        state->gated,
        key_heads,
        value_heads,
        k_dim,
        v_dim,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent 失败");
    launch_float_to_lowp(state->gated, state->gated_bf16, value_total, out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 linear gated 失败");

    const int out_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(out_dim) * sizeof(float), "linear out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->gated_bf16, out_device->type, cache.out_buffer);
    out.assign(static_cast<size_t>(out_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, static_cast<size_t>(out_dim) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy linear out 失败");
    return true;
#else
    (void) in_proj_qkv_w;
    (void) in_proj_z_w;
    (void) in_proj_b_w;
    (void) in_proj_a_w;
    (void) conv_w;
    (void) a_log;
    (void) dt_bias;
    (void) norm_w;
    (void) out_w;
    (void) x;
    (void) state_handle;
    (void) key_heads;
    (void) value_heads;
    (void) k_dim;
    (void) v_dim;
    (void) kernel;
    (void) eps;
    (void) out;
    return false;
#endif
}

bool cuda_rmsnorm_linear_attention_project_layer(
    const TensorRef & input_norm_w,
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state_handle,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    bool one_plus,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const uint16_t * device_x = cuda_rms_norm_input_to_bf16(input_norm_w, x, eps, one_plus);
    if (!device_x) {
        return false;
    }
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    if (in_proj_qkv_w.info->dtype != "BF16" ||
        in_proj_z_w.info->dtype != "BF16" ||
        in_proj_b_w.info->dtype != "BF16" ||
        in_proj_a_w.info->dtype != "BF16" ||
        conv_w.info->dtype != "BF16" ||
        a_log.info->dtype != "F32" ||
        dt_bias.info->dtype != "BF16" ||
        norm_w.info->dtype != "F32" ||
        out_w.info->dtype != "BF16" ||
        in_proj_qkv_w.info->shape[0] != conv_dim ||
        in_proj_qkv_w.info->shape[1] != static_cast<int64_t>(x.size()) ||
        in_proj_z_w.info->shape[0] != value_total ||
        in_proj_b_w.info->shape[0] != value_heads ||
        in_proj_a_w.info->shape[0] != value_heads) {
        return false;
    }

    DeviceWeight * projection_device = cached_cuda_concat_weight(
        in_proj_qkv_w.info->name + "\n" + in_proj_z_w.info->name + "\n" + in_proj_b_w.info->name + "\n" + in_proj_a_w.info->name,
        {in_proj_qkv_w, in_proj_z_w, in_proj_b_w, in_proj_a_w});
    DeviceWeight * conv_device = cached_cuda_weight(conv_w);
    DeviceWeight * a_log_device = cached_cuda_weight(a_log);
    DeviceWeight * dt_bias_device = cached_cuda_weight(dt_bias);
    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!projection_device || !conv_device || !a_log_device || !dt_bias_device || !norm_device || !out_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    CudaLinearAttentionState * state =
        ensure_linear_attention_state(state_handle, key_heads, value_heads, k_dim, v_dim, kernel);
    TensorInfo combined_info = *in_proj_qkv_w.info;
    combined_info.name = in_proj_qkv_w.info->name + "+z+b+a";
    combined_info.shape[0] = static_cast<int64_t>(conv_dim + value_total + value_heads * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, device_x, CUDA_R_16BF, state->projection);
    const float * mixed_ptr = state->projection;
    const float * z_ptr = state->projection + conv_dim;
    const float * b_ptr = state->projection + conv_dim + value_total;
    const float * a_ptr = state->projection + conv_dim + value_total + value_heads;

    launch_linear_attention_conv(
        mixed_ptr,
        static_cast<const uint16_t *>(conv_device->ptr),
        state->conv_state,
        state->conv_out,
        conv_dim,
        kernel,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_conv 失败");
    launch_linear_attention_recurrent(
        state->conv_out,
        z_ptr,
        b_ptr,
        a_ptr,
        static_cast<const float *>(a_log_device->ptr),
        static_cast<const uint16_t *>(dt_bias_device->ptr),
        static_cast<const float *>(norm_device->ptr),
        state->recurrent_state,
        state->gated,
        key_heads,
        value_heads,
        k_dim,
        v_dim,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent 失败");
    launch_float_to_lowp(state->gated, state->gated_bf16, value_total, out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 linear gated 失败");

    const int out_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(out_dim) * sizeof(float), "linear out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->gated_bf16, out_device->type, cache.out_buffer);
    out.assign(static_cast<size_t>(out_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, static_cast<size_t>(out_dim) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy linear out 失败");
    return true;
#else
    (void) input_norm_w;
    (void) in_proj_qkv_w;
    (void) in_proj_z_w;
    (void) in_proj_b_w;
    (void) in_proj_a_w;
    (void) conv_w;
    (void) a_log;
    (void) dt_bias;
    (void) norm_w;
    (void) out_w;
    (void) x;
    (void) state_handle;
    (void) key_heads;
    (void) value_heads;
    (void) k_dim;
    (void) v_dim;
    (void) kernel;
    (void) eps;
    (void) one_plus;
    (void) out;
    return false;
#endif
}

bool cuda_linear_attention_full_layer(
    const TensorRef & input_norm_w,
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & attn_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const std::vector<float> & x,
    void *& state_handle,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    bool one_plus,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const int hidden_dim = static_cast<int>(x.size());
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const uint16_t * normed_x = cuda_rms_norm_input_to_bf16(input_norm_w, x, eps, one_plus);
    if (!normed_x) {
        return false;
    }
    DeviceWeight * projection_device = cached_cuda_concat_weight(
        in_proj_qkv_w.info->name + "\n" + in_proj_z_w.info->name + "\n" + in_proj_b_w.info->name + "\n" + in_proj_a_w.info->name,
        {in_proj_qkv_w, in_proj_z_w, in_proj_b_w, in_proj_a_w});
    DeviceWeight * conv_device = cached_cuda_weight(conv_w);
    DeviceWeight * a_log_device = cached_cuda_weight(a_log);
    DeviceWeight * dt_bias_device = cached_cuda_weight(dt_bias);
    DeviceWeight * attn_norm_device = cached_cuda_weight(attn_norm_w);
    DeviceWeight * attn_out_device = cached_cuda_matvec_weight(attn_out_w);
    DeviceWeight * post_norm_device = cached_cuda_weight(post_norm_w);
    if (!projection_device || !conv_device || !a_log_device || !dt_bias_device || !attn_norm_device || !attn_out_device || !post_norm_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t hidden_float_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t hidden_bf16_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_float_buffer(cache.residual_buffer, cache.residual_bytes, hidden_float_bytes, "layer residual buffer");
    ensure_float_buffer(cache.mixer_buffer, cache.mixer_bytes, hidden_float_bytes, "layer mixer buffer");
    ensure_float_buffer(cache.layer_out_buffer, cache.layer_out_bytes, hidden_float_bytes, "layer out buffer");
    ensure_float_buffer(cache.mlp_out_buffer, cache.mlp_out_bytes, hidden_float_bytes, "layer mlp out buffer");
    ensure_u16_buffer(cache.post_norm_bf16_buffer, cache.post_norm_bf16_bytes, hidden_bf16_bytes, "post norm bf16 buffer");
    check_cuda(cudaMemcpy(cache.residual_buffer, x.data(), hidden_float_bytes, cudaMemcpyHostToDevice), "cudaMemcpy layer residual 失败");

    CudaLinearAttentionState * state =
        ensure_linear_attention_state(state_handle, key_heads, value_heads, k_dim, v_dim, kernel);
    TensorInfo combined_info = *in_proj_qkv_w.info;
    combined_info.name = in_proj_qkv_w.info->name + "+z+b+a";
    combined_info.shape[0] = static_cast<int64_t>(conv_dim + value_total + value_heads * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, normed_x, CUDA_R_16BF, state->projection);
    const float * mixed_ptr = state->projection;
    const float * z_ptr = state->projection + conv_dim;
    const float * b_ptr = state->projection + conv_dim + value_total;
    const float * a_ptr = state->projection + conv_dim + value_total + value_heads;

    launch_linear_attention_conv(mixed_ptr, static_cast<const uint16_t *>(conv_device->ptr), state->conv_state, state->conv_out, conv_dim, kernel, nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_conv 失败");
    launch_linear_attention_recurrent(
        state->conv_out,
        z_ptr,
        b_ptr,
        a_ptr,
        static_cast<const float *>(a_log_device->ptr),
        static_cast<const uint16_t *>(dt_bias_device->ptr),
        static_cast<const float *>(attn_norm_device->ptr),
        state->recurrent_state,
        state->gated,
        key_heads,
        value_heads,
        k_dim,
        v_dim,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent 失败");
    launch_float_to_lowp(state->gated, state->gated_bf16, value_total, attn_out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 linear gated 失败");
    cublas_matvec_to_device(cache, attn_out_w, *attn_out_device, state->gated_bf16, attn_out_device->type, cache.mixer_buffer);

    launch_add_rms_norm_to_bf16(
        cache.residual_buffer,
        cache.mixer_buffer,
        static_cast<const uint16_t *>(post_norm_device->ptr),
        cache.layer_out_buffer,
        cache.post_norm_bf16_buffer,
        hidden_dim,
        eps,
        one_plus,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_add_rms_norm_to_bf16 post 失败");
    if (!cuda_mlp_from_device_bf16_to_device(mlp_gate_w, mlp_up_w, mlp_down_w, cache.post_norm_bf16_buffer, cache.mlp_out_buffer)) {
        return false;
    }
    launch_add_float(cache.layer_out_buffer, cache.mlp_out_buffer, cache.layer_out_buffer, hidden_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_add_float mlp residual 失败");

    out.assign(static_cast<size_t>(hidden_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.layer_out_buffer, hidden_float_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy layer out 失败");
    return true;
#else
    (void) input_norm_w;
    (void) in_proj_qkv_w;
    (void) in_proj_z_w;
    (void) in_proj_b_w;
    (void) in_proj_a_w;
    (void) conv_w;
    (void) a_log;
    (void) dt_bias;
    (void) attn_norm_w;
    (void) attn_out_w;
    (void) post_norm_w;
    (void) mlp_gate_w;
    (void) mlp_up_w;
    (void) mlp_down_w;
    (void) x;
    (void) state_handle;
    (void) key_heads;
    (void) value_heads;
    (void) k_dim;
    (void) v_dim;
    (void) kernel;
    (void) eps;
    (void) one_plus;
    (void) out;
    return false;
#endif
}

bool cuda_linear_attention_full_layer_device(
    const TensorRef & input_norm_w,
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & attn_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const void * device_x,
    void * device_out,
    int hidden_dim,
    void *& state_handle,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    bool one_plus) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (!device_x || !device_out || input_norm_w.info->dtype != "BF16") {
        return false;
    }
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    DeviceWeight * input_norm_device = cached_cuda_weight(input_norm_w);
    DeviceWeight * projection_device = cached_cuda_concat_weight(
        in_proj_qkv_w.info->name + "\n" + in_proj_z_w.info->name + "\n" + in_proj_b_w.info->name + "\n" + in_proj_a_w.info->name,
        {in_proj_qkv_w, in_proj_z_w, in_proj_b_w, in_proj_a_w});
    DeviceWeight * conv_device = cached_cuda_weight(conv_w);
    DeviceWeight * a_log_device = cached_cuda_weight(a_log);
    DeviceWeight * dt_bias_device = cached_cuda_weight(dt_bias);
    DeviceWeight * attn_norm_device = cached_cuda_weight(attn_norm_w);
    DeviceWeight * attn_out_device = cached_cuda_weight(attn_out_w);
    DeviceWeight * post_norm_device = cached_cuda_weight(post_norm_w);
    if (!input_norm_device || !projection_device || !conv_device || !a_log_device || !dt_bias_device || !attn_norm_device || !attn_out_device || !post_norm_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t hidden_float_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t hidden_bf16_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_float_buffer(cache.mixer_buffer, cache.mixer_bytes, hidden_float_bytes, "layer mixer buffer");
    ensure_float_buffer(cache.layer_out_buffer, cache.layer_out_bytes, hidden_float_bytes, "layer out buffer");
    ensure_float_buffer(cache.mlp_out_buffer, cache.mlp_out_bytes, hidden_float_bytes, "layer mlp out buffer");
    ensure_u16_buffer(cache.norm_bf16_buffer, cache.norm_bf16_bytes, hidden_bf16_bytes, "input norm bf16 buffer");
    ensure_u16_buffer(cache.post_norm_bf16_buffer, cache.post_norm_bf16_bytes, hidden_bf16_bytes, "post norm bf16 buffer");

    if (projection_device->type == CUDA_R_16F) {
        launch_rms_norm_to_f16(
            static_cast<const float *>(device_x),
            static_cast<const uint16_t *>(input_norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_dim,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_f16 device linear input 失败");
    } else {
        launch_rms_norm_to_bf16(
            static_cast<const float *>(device_x),
            static_cast<const uint16_t *>(input_norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_dim,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 device linear input 失败");
    }

    CudaLinearAttentionState * state =
        ensure_linear_attention_state(state_handle, key_heads, value_heads, k_dim, v_dim, kernel);
    TensorInfo combined_info = *in_proj_qkv_w.info;
    combined_info.name = in_proj_qkv_w.info->name + "+z+b+a";
    combined_info.shape[0] = static_cast<int64_t>(conv_dim + value_total + value_heads * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, cache.norm_bf16_buffer, projection_device->type, state->projection);
    const float * mixed_ptr = state->projection;
    const float * z_ptr = state->projection + conv_dim;
    const float * b_ptr = state->projection + conv_dim + value_total;
    const float * a_ptr = state->projection + conv_dim + value_total + value_heads;

    launch_linear_attention_conv(mixed_ptr, static_cast<const uint16_t *>(conv_device->ptr), state->conv_state, state->conv_out, conv_dim, kernel, nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_conv device 失败");
    launch_linear_attention_recurrent(
        state->conv_out,
        z_ptr,
        b_ptr,
        a_ptr,
        static_cast<const float *>(a_log_device->ptr),
        static_cast<const uint16_t *>(dt_bias_device->ptr),
        static_cast<const float *>(attn_norm_device->ptr),
        state->recurrent_state,
        state->gated,
        key_heads,
        value_heads,
        k_dim,
        v_dim,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent device 失败");
    launch_float_to_lowp(state->gated, state->gated_bf16, value_total, attn_out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_lowp linear gated device 失败");
    cublas_matvec_to_device(cache, attn_out_w, *attn_out_device, state->gated_bf16, attn_out_device->type, cache.mixer_buffer);

    launch_add_rms_norm_to_bf16(
        static_cast<const float *>(device_x),
        cache.mixer_buffer,
        static_cast<const uint16_t *>(post_norm_device->ptr),
        cache.layer_out_buffer,
        cache.post_norm_bf16_buffer,
        hidden_dim,
        eps,
        one_plus,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_add_rms_norm_to_bf16 linear device post 失败");
    if (!cuda_mlp_from_device_bf16_to_device(mlp_gate_w, mlp_up_w, mlp_down_w, cache.post_norm_bf16_buffer, cache.mlp_out_buffer)) {
        return false;
    }
    launch_add_float(cache.layer_out_buffer, cache.mlp_out_buffer, static_cast<float *>(device_out), hidden_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_add_float linear device mlp residual 失败");
    return true;
#else
    (void) input_norm_w;
    (void) in_proj_qkv_w;
    (void) in_proj_z_w;
    (void) in_proj_b_w;
    (void) in_proj_a_w;
    (void) conv_w;
    (void) a_log;
    (void) dt_bias;
    (void) attn_norm_w;
    (void) attn_out_w;
    (void) post_norm_w;
    (void) mlp_gate_w;
    (void) mlp_up_w;
    (void) mlp_down_w;
    (void) device_x;
    (void) device_out;
    (void) hidden_dim;
    (void) state_handle;
    (void) key_heads;
    (void) value_heads;
    (void) k_dim;
    (void) v_dim;
    (void) kernel;
    (void) eps;
    (void) one_plus;
    return false;
#endif
}

bool cuda_full_attention_layer(
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & out_w,
    const std::vector<float> & q_and_gate,
    const std::vector<float> & k,
    const std::vector<float> & v,
    void *& state_handle,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    if (q_norm_w.info->dtype != "BF16" || k_norm_w.info->dtype != "BF16" || out_w.info->dtype != "BF16") {
        return false;
    }
    if (q_norm_w.info->shape.size() != 1 ||
        q_norm_w.info->shape[0] != head_dim ||
        k_norm_w.info->shape.size() != 1 ||
        k_norm_w.info->shape[0] != head_dim ||
        out_w.info->shape.size() != 2 ||
        out_w.info->shape[1] != q_total ||
        static_cast<int>(q_and_gate.size()) != q_total * 2 ||
        static_cast<int>(k.size()) != kv_total ||
        static_cast<int>(v.size()) != kv_total ||
        pos < 0 ||
        pos >= max_seq_len) {
        return false;
    }

    DeviceWeight * q_norm_device = cached_cuda_weight(q_norm_w);
    DeviceWeight * k_norm_device = cached_cuda_weight(k_norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!q_norm_device || !k_norm_device || !out_device) {
        return false;
    }

    CudaFullAttentionState * state =
        ensure_full_attention_state(state_handle, n_heads, kv_heads, head_dim, max_seq_len);
    check_cuda(cudaMemcpy(state->q_and_gate, q_and_gate.data(), static_cast<size_t>(q_total) * 2 * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy full q_and_gate 失败");
    check_cuda(cudaMemcpy(state->k, k.data(), static_cast<size_t>(kv_total) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy full k 失败");
    check_cuda(cudaMemcpy(state->v, v.data(), static_cast<size_t>(kv_total) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy full v 失败");

    launch_full_attention_q(
        state->q_and_gate,
        static_cast<const uint16_t *>(q_norm_device->ptr),
        state->q,
        state->gate,
        n_heads,
        head_dim,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_q 失败");
    launch_full_attention_kv(
        state->k,
        state->v,
        static_cast<const uint16_t *>(k_norm_device->ptr),
        state->key_cache,
        state->value_cache,
        kv_heads,
        head_dim,
        max_seq_len,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_kv 失败");
    launch_full_attention_attend(
        state->q,
        state->gate,
        state->key_cache,
        state->value_cache,
        state->attn,
        n_heads,
        kv_heads,
        head_dim,
        max_seq_len,
        pos,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_attend 失败");
    launch_float_to_bf16(state->attn, state->attn_bf16, q_total, nullptr);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 full attn 失败");

    auto & cache = cuda_weight_cache();
    const int hidden_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(hidden_dim) * sizeof(float), "full out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->attn_bf16, CUDA_R_16BF, cache.out_buffer);
    out.assign(static_cast<size_t>(hidden_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, static_cast<size_t>(hidden_dim) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy full out 失败");
    return true;
#else
    (void) q_norm_w;
    (void) k_norm_w;
    (void) out_w;
    (void) q_and_gate;
    (void) k;
    (void) v;
    (void) state_handle;
    (void) n_heads;
    (void) kv_heads;
    (void) head_dim;
    (void) max_seq_len;
    (void) pos;
    (void) rope_theta;
    (void) partial_rotary_factor;
    (void) eps;
    (void) out;
    return false;
#endif
}

bool cuda_full_attention_project_layer(
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state_handle,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    if (q_proj_w.info->dtype != "BF16" ||
        k_proj_w.info->dtype != "BF16" ||
        v_proj_w.info->dtype != "BF16" ||
        q_norm_w.info->dtype != "BF16" ||
        k_norm_w.info->dtype != "BF16" ||
        out_w.info->dtype != "BF16") {
        return false;
    }
    if (static_cast<int>(x.size()) != q_proj_w.info->shape[1] ||
        q_proj_w.info->shape[0] != q_total * 2 ||
        k_proj_w.info->shape[0] != kv_total ||
        v_proj_w.info->shape[0] != kv_total ||
        out_w.info->shape[1] != q_total ||
        pos < 0 ||
        pos >= max_seq_len) {
        return false;
    }

    DeviceWeight * projection_device = cached_cuda_matvec_concat_weight(
        q_proj_w.info->name + "\n" + k_proj_w.info->name + "\n" + v_proj_w.info->name,
        {q_proj_w, k_proj_w, v_proj_w});
    DeviceWeight * q_norm_device = cached_cuda_weight(q_norm_w);
    DeviceWeight * k_norm_device = cached_cuda_weight(k_norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!projection_device || !q_norm_device || !k_norm_device || !out_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const int hidden_dim = static_cast<int>(x.size());
    const size_t x_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_cuda_buffers(cache, x_bytes, 1);
    std::vector<uint16_t> x_lowp = host_float_to_lowp(x, projection_device->type);
    check_cuda(cudaMemcpy(cache.x_buffer, x_lowp.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy full project x 失败");

    CudaFullAttentionState * state =
        ensure_full_attention_state(state_handle, n_heads, kv_heads, head_dim, max_seq_len);
    TensorInfo combined_info = *q_proj_w.info;
    combined_info.name = q_proj_w.info->name + "+k+v";
    combined_info.shape[0] = static_cast<int64_t>(q_total * 2 + kv_total * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, cache.x_buffer, projection_device->type, state->projection);
    const float * q_and_gate_ptr = state->projection;
    const float * k_ptr = state->projection + q_total * 2;
    const float * v_ptr = state->projection + q_total * 2 + kv_total;

    launch_full_attention_q(
        q_and_gate_ptr,
        static_cast<const uint16_t *>(q_norm_device->ptr),
        state->q,
        state->gate,
        n_heads,
        head_dim,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_q 失败");
    launch_full_attention_kv(
        k_ptr,
        v_ptr,
        static_cast<const uint16_t *>(k_norm_device->ptr),
        state->key_cache,
        state->value_cache,
        kv_heads,
        head_dim,
        max_seq_len,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_kv 失败");
    launch_full_attention_attend(
        state->q,
        state->gate,
        state->key_cache,
        state->value_cache,
        state->attn,
        n_heads,
        kv_heads,
        head_dim,
        max_seq_len,
        pos,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_attend 失败");
    launch_float_to_lowp(state->attn, state->attn_bf16, q_total, out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 full attn 失败");

    const int out_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(out_dim) * sizeof(float), "full out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->attn_bf16, out_device->type, cache.out_buffer);
    out.assign(static_cast<size_t>(out_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, static_cast<size_t>(out_dim) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy full out 失败");
    return true;
#else
    (void) q_proj_w;
    (void) k_proj_w;
    (void) v_proj_w;
    (void) q_norm_w;
    (void) k_norm_w;
    (void) out_w;
    (void) x;
    (void) state_handle;
    (void) n_heads;
    (void) kv_heads;
    (void) head_dim;
    (void) max_seq_len;
    (void) pos;
    (void) rope_theta;
    (void) partial_rotary_factor;
    (void) eps;
    (void) out;
    return false;
#endif
}

bool cuda_rmsnorm_full_attention_project_layer(
    const TensorRef & input_norm_w,
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state_handle,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    bool one_plus,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const uint16_t * device_x = cuda_rms_norm_input_to_bf16(input_norm_w, x, eps, one_plus);
    if (!device_x) {
        return false;
    }
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    if (q_proj_w.info->dtype != "BF16" ||
        k_proj_w.info->dtype != "BF16" ||
        v_proj_w.info->dtype != "BF16" ||
        q_norm_w.info->dtype != "BF16" ||
        k_norm_w.info->dtype != "BF16" ||
        out_w.info->dtype != "BF16" ||
        q_proj_w.info->shape[1] != static_cast<int64_t>(x.size()) ||
        q_proj_w.info->shape[0] != q_total * 2 ||
        k_proj_w.info->shape[0] != kv_total ||
        v_proj_w.info->shape[0] != kv_total ||
        out_w.info->shape[1] != q_total ||
        pos < 0 ||
        pos >= max_seq_len) {
        return false;
    }

    DeviceWeight * projection_device = cached_cuda_concat_weight(
        q_proj_w.info->name + "\n" + k_proj_w.info->name + "\n" + v_proj_w.info->name,
        {q_proj_w, k_proj_w, v_proj_w});
    DeviceWeight * q_norm_device = cached_cuda_weight(q_norm_w);
    DeviceWeight * k_norm_device = cached_cuda_weight(k_norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!projection_device || !q_norm_device || !k_norm_device || !out_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    CudaFullAttentionState * state =
        ensure_full_attention_state(state_handle, n_heads, kv_heads, head_dim, max_seq_len);
    TensorInfo combined_info = *q_proj_w.info;
    combined_info.name = q_proj_w.info->name + "+k+v";
    combined_info.shape[0] = static_cast<int64_t>(q_total * 2 + kv_total * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, device_x, CUDA_R_16BF, state->projection);
    const float * q_and_gate_ptr = state->projection;
    const float * k_ptr = state->projection + q_total * 2;
    const float * v_ptr = state->projection + q_total * 2 + kv_total;

    launch_full_attention_q(
        q_and_gate_ptr,
        static_cast<const uint16_t *>(q_norm_device->ptr),
        state->q,
        state->gate,
        n_heads,
        head_dim,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_q 失败");
    launch_full_attention_kv(
        k_ptr,
        v_ptr,
        static_cast<const uint16_t *>(k_norm_device->ptr),
        state->key_cache,
        state->value_cache,
        kv_heads,
        head_dim,
        max_seq_len,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_kv 失败");
    launch_full_attention_attend(
        state->q,
        state->gate,
        state->key_cache,
        state->value_cache,
        state->attn,
        n_heads,
        kv_heads,
        head_dim,
        max_seq_len,
        pos,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_attend 失败");
    launch_float_to_lowp(state->attn, state->attn_bf16, q_total, out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 full attn 失败");

    const int out_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(out_dim) * sizeof(float), "full out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->attn_bf16, out_device->type, cache.out_buffer);
    out.assign(static_cast<size_t>(out_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, static_cast<size_t>(out_dim) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy full out 失败");
    return true;
#else
    (void) input_norm_w;
    (void) q_proj_w;
    (void) k_proj_w;
    (void) v_proj_w;
    (void) q_norm_w;
    (void) k_norm_w;
    (void) out_w;
    (void) x;
    (void) state_handle;
    (void) n_heads;
    (void) kv_heads;
    (void) head_dim;
    (void) max_seq_len;
    (void) pos;
    (void) rope_theta;
    (void) partial_rotary_factor;
    (void) eps;
    (void) one_plus;
    (void) out;
    return false;
#endif
}

bool cuda_full_attention_full_layer(
    const TensorRef & input_norm_w,
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const std::vector<float> & x,
    void *& state_handle,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    bool one_plus,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    const int hidden_dim = static_cast<int>(x.size());
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const uint16_t * normed_x = cuda_rms_norm_input_to_bf16(input_norm_w, x, eps, one_plus);
    if (!normed_x) {
        return false;
    }
    DeviceWeight * projection_device = cached_cuda_concat_weight(
        q_proj_w.info->name + "\n" + k_proj_w.info->name + "\n" + v_proj_w.info->name,
        {q_proj_w, k_proj_w, v_proj_w});
    DeviceWeight * q_norm_device = cached_cuda_weight(q_norm_w);
    DeviceWeight * k_norm_device = cached_cuda_weight(k_norm_w);
    DeviceWeight * attn_out_device = cached_cuda_matvec_weight(attn_out_w);
    DeviceWeight * post_norm_device = cached_cuda_weight(post_norm_w);
    if (!projection_device || !q_norm_device || !k_norm_device || !attn_out_device || !post_norm_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t hidden_float_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t hidden_bf16_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_float_buffer(cache.residual_buffer, cache.residual_bytes, hidden_float_bytes, "layer residual buffer");
    ensure_float_buffer(cache.mixer_buffer, cache.mixer_bytes, hidden_float_bytes, "layer mixer buffer");
    ensure_float_buffer(cache.layer_out_buffer, cache.layer_out_bytes, hidden_float_bytes, "layer out buffer");
    ensure_float_buffer(cache.mlp_out_buffer, cache.mlp_out_bytes, hidden_float_bytes, "layer mlp out buffer");
    ensure_u16_buffer(cache.post_norm_bf16_buffer, cache.post_norm_bf16_bytes, hidden_bf16_bytes, "post norm bf16 buffer");
    check_cuda(cudaMemcpy(cache.residual_buffer, x.data(), hidden_float_bytes, cudaMemcpyHostToDevice), "cudaMemcpy layer residual 失败");

    CudaFullAttentionState * state =
        ensure_full_attention_state(state_handle, n_heads, kv_heads, head_dim, max_seq_len);
    TensorInfo combined_info = *q_proj_w.info;
    combined_info.name = q_proj_w.info->name + "+k+v";
    combined_info.shape[0] = static_cast<int64_t>(q_total * 2 + kv_total * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, normed_x, CUDA_R_16BF, state->projection);
    const float * q_and_gate_ptr = state->projection;
    const float * k_ptr = state->projection + q_total * 2;
    const float * v_ptr = state->projection + q_total * 2 + kv_total;

    launch_full_attention_q(q_and_gate_ptr, static_cast<const uint16_t *>(q_norm_device->ptr), state->q, state->gate, n_heads, head_dim, pos, rope_theta, partial_rotary_factor, eps, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_q 失败");
    launch_full_attention_kv(k_ptr, v_ptr, static_cast<const uint16_t *>(k_norm_device->ptr), state->key_cache, state->value_cache, kv_heads, head_dim, max_seq_len, pos, rope_theta, partial_rotary_factor, eps, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_kv 失败");
    launch_full_attention_attend(state->q, state->gate, state->key_cache, state->value_cache, state->attn, n_heads, kv_heads, head_dim, max_seq_len, pos, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_attend 失败");
    launch_float_to_lowp(state->attn, state->attn_bf16, q_total, attn_out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 full attn 失败");
    cublas_matvec_to_device(cache, attn_out_w, *attn_out_device, state->attn_bf16, attn_out_device->type, cache.mixer_buffer);

    launch_add_rms_norm_to_bf16(
        cache.residual_buffer,
        cache.mixer_buffer,
        static_cast<const uint16_t *>(post_norm_device->ptr),
        cache.layer_out_buffer,
        cache.post_norm_bf16_buffer,
        hidden_dim,
        eps,
        one_plus,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_add_rms_norm_to_bf16 post 失败");
    if (!cuda_mlp_from_device_bf16_to_device(mlp_gate_w, mlp_up_w, mlp_down_w, cache.post_norm_bf16_buffer, cache.mlp_out_buffer)) {
        return false;
    }
    launch_add_float(cache.layer_out_buffer, cache.mlp_out_buffer, cache.layer_out_buffer, hidden_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_add_float mlp residual 失败");

    out.assign(static_cast<size_t>(hidden_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.layer_out_buffer, hidden_float_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy layer out 失败");
    return true;
#else
    (void) input_norm_w;
    (void) q_proj_w;
    (void) k_proj_w;
    (void) v_proj_w;
    (void) q_norm_w;
    (void) k_norm_w;
    (void) attn_out_w;
    (void) post_norm_w;
    (void) mlp_gate_w;
    (void) mlp_up_w;
    (void) mlp_down_w;
    (void) x;
    (void) state_handle;
    (void) n_heads;
    (void) kv_heads;
    (void) head_dim;
    (void) max_seq_len;
    (void) pos;
    (void) rope_theta;
    (void) partial_rotary_factor;
    (void) eps;
    (void) one_plus;
    (void) out;
    return false;
#endif
}

bool cuda_full_attention_full_layer_device(
    const TensorRef & input_norm_w,
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const void * device_x,
    void * device_out,
    int hidden_dim,
    void *& state_handle,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    bool one_plus) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (!device_x || !device_out || input_norm_w.info->dtype != "BF16") {
        return false;
    }
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    DeviceWeight * input_norm_device = cached_cuda_weight(input_norm_w);
    DeviceWeight * projection_device = cached_cuda_concat_weight(
        q_proj_w.info->name + "\n" + k_proj_w.info->name + "\n" + v_proj_w.info->name,
        {q_proj_w, k_proj_w, v_proj_w});
    DeviceWeight * q_norm_device = cached_cuda_weight(q_norm_w);
    DeviceWeight * k_norm_device = cached_cuda_weight(k_norm_w);
    DeviceWeight * attn_out_device = cached_cuda_weight(attn_out_w);
    DeviceWeight * post_norm_device = cached_cuda_weight(post_norm_w);
    if (!input_norm_device || !projection_device || !q_norm_device || !k_norm_device || !attn_out_device || !post_norm_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t hidden_float_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t hidden_bf16_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_float_buffer(cache.mixer_buffer, cache.mixer_bytes, hidden_float_bytes, "layer mixer buffer");
    ensure_float_buffer(cache.layer_out_buffer, cache.layer_out_bytes, hidden_float_bytes, "layer out buffer");
    ensure_float_buffer(cache.mlp_out_buffer, cache.mlp_out_bytes, hidden_float_bytes, "layer mlp out buffer");
    ensure_u16_buffer(cache.norm_bf16_buffer, cache.norm_bf16_bytes, hidden_bf16_bytes, "input norm bf16 buffer");
    ensure_u16_buffer(cache.post_norm_bf16_buffer, cache.post_norm_bf16_bytes, hidden_bf16_bytes, "post norm bf16 buffer");

    if (projection_device->type == CUDA_R_16F) {
        launch_rms_norm_to_f16(
            static_cast<const float *>(device_x),
            static_cast<const uint16_t *>(input_norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_dim,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_f16 device full input 失败");
    } else {
        launch_rms_norm_to_bf16(
            static_cast<const float *>(device_x),
            static_cast<const uint16_t *>(input_norm_device->ptr),
            cache.norm_bf16_buffer,
            hidden_dim,
            eps,
            one_plus,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 device full input 失败");
    }

    CudaFullAttentionState * state =
        ensure_full_attention_state(state_handle, n_heads, kv_heads, head_dim, max_seq_len);
    TensorInfo combined_info = *q_proj_w.info;
    combined_info.name = q_proj_w.info->name + "+k+v";
    combined_info.shape[0] = static_cast<int64_t>(q_total * 2 + kv_total * 2);
    TensorRef combined_ref {&combined_info, nullptr};
    cublas_matvec_to_device(cache, combined_ref, *projection_device, cache.norm_bf16_buffer, projection_device->type, state->projection);
    const float * q_and_gate_ptr = state->projection;
    const float * k_ptr = state->projection + q_total * 2;
    const float * v_ptr = state->projection + q_total * 2 + kv_total;

    launch_full_attention_q(q_and_gate_ptr, static_cast<const uint16_t *>(q_norm_device->ptr), state->q, state->gate, n_heads, head_dim, pos, rope_theta, partial_rotary_factor, eps, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_q device 失败");
    launch_full_attention_kv(k_ptr, v_ptr, static_cast<const uint16_t *>(k_norm_device->ptr), state->key_cache, state->value_cache, kv_heads, head_dim, max_seq_len, pos, rope_theta, partial_rotary_factor, eps, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_kv device 失败");
    launch_full_attention_attend(state->q, state->gate, state->key_cache, state->value_cache, state->attn, n_heads, kv_heads, head_dim, max_seq_len, pos, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_attend device 失败");
    launch_float_to_lowp(state->attn, state->attn_bf16, q_total, attn_out_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_lowp full attn device 失败");
    cublas_matvec_to_device(cache, attn_out_w, *attn_out_device, state->attn_bf16, attn_out_device->type, cache.mixer_buffer);

    launch_add_rms_norm_to_bf16(
        static_cast<const float *>(device_x),
        cache.mixer_buffer,
        static_cast<const uint16_t *>(post_norm_device->ptr),
        cache.layer_out_buffer,
        cache.post_norm_bf16_buffer,
        hidden_dim,
        eps,
        one_plus,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_add_rms_norm_to_bf16 full device post 失败");
    if (!cuda_mlp_from_device_bf16_to_device(mlp_gate_w, mlp_up_w, mlp_down_w, cache.post_norm_bf16_buffer, cache.mlp_out_buffer)) {
        return false;
    }
    launch_add_float(cache.layer_out_buffer, cache.mlp_out_buffer, static_cast<float *>(device_out), hidden_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_add_float full device mlp residual 失败");
    return true;
#else
    (void) input_norm_w;
    (void) q_proj_w;
    (void) k_proj_w;
    (void) v_proj_w;
    (void) q_norm_w;
    (void) k_norm_w;
    (void) attn_out_w;
    (void) post_norm_w;
    (void) mlp_gate_w;
    (void) mlp_up_w;
    (void) mlp_down_w;
    (void) device_x;
    (void) device_out;
    (void) hidden_dim;
    (void) state_handle;
    (void) n_heads;
    (void) kv_heads;
    (void) head_dim;
    (void) max_seq_len;
    (void) pos;
    (void) rope_theta;
    (void) partial_rotary_factor;
    (void) eps;
    (void) one_plus;
    return false;
#endif
}

const void * cuda_prefill_batch(
    const ModelConfig & config,
    const ModelWeights & weights,
    const std::vector<int> & prompt_ids,
    std::vector<void *> & linear_states,
    std::vector<void *> & full_states,
    const std::vector<int> & full_max_seq_lens,
    int & seq_len) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (prompt_ids.empty() || seq_len != 0) {
        return nullptr;
    }
    const int tokens = static_cast<int>(prompt_ids.size());
    const int hidden_dim = config.text.hidden_size;
    auto tensor = [&](const std::string & name) {
        return weights.tensor_ref(name);
    };

    auto & cache = cuda_weight_cache();
    float * current = static_cast<float *>(cuda_token_hidden_buffer(0, tokens * hidden_dim));
    float * next = static_cast<float *>(cuda_token_hidden_buffer(1, tokens * hidden_dim));
    int * token_ids = static_cast<int *>(cuda_generated_token_buffer(tokens));
    if (!current || !next || !token_ids) {
        return nullptr;
    }
    check_cuda(cudaMemcpy(token_ids, prompt_ids.data(), static_cast<size_t>(tokens) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy prompt ids 失败");
    const TensorRef emb = tensor("model.language_model.embed_tokens.weight");
    DeviceWeight * emb_device = cached_cuda_weight(emb);
    if (!emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        return nullptr;
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
    ensure_float_buffer(cache.mixer_buffer, cache.mixer_bytes, hidden_float_bytes, "batch mixer");
    ensure_float_buffer(cache.layer_out_buffer, cache.layer_out_bytes, hidden_float_bytes, "batch layer out");
    ensure_float_buffer(cache.mlp_out_buffer, cache.mlp_out_bytes, hidden_float_bytes, "batch mlp out");
    ensure_u16_buffer(cache.norm_bf16_buffer, cache.norm_bf16_bytes, hidden_bf16_bytes, "batch norm bf16");
    ensure_u16_buffer(cache.post_norm_bf16_buffer, cache.post_norm_bf16_bytes, hidden_bf16_bytes, "batch post norm bf16");

    for (int layer = 0; layer < config.text.num_hidden_layers; ++layer) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
        const TensorRef input_norm_w = tensor(prefix + "input_layernorm.weight");
        DeviceWeight * input_norm_device = cached_cuda_weight(input_norm_w);
        if (!input_norm_device) {
            return nullptr;
        }
        launch_rms_norm_batch_to_bf16(
            current,
            static_cast<const uint16_t *>(input_norm_device->ptr),
            cache.norm_bf16_buffer,
            tokens,
            hidden_dim,
            config.text.rms_norm_eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_batch_to_bf16 input 失败");

        if (config.text.layer_types[layer] == "linear_attention") {
            const int key_heads = config.text.linear_num_key_heads;
            const int value_heads = config.text.linear_num_value_heads;
            const int k_dim = config.text.linear_key_head_dim;
            const int v_dim = config.text.linear_value_head_dim;
            const int key_total = key_heads * k_dim;
            const int value_total = value_heads * v_dim;
            const int conv_dim = key_total * 2 + value_total;
            CudaLinearAttentionState * state =
                ensure_linear_attention_state(linear_states[layer], key_heads, value_heads, k_dim, v_dim, config.text.linear_conv_kernel_dim);
            ensure_float_buffer(state->batch_projection, state->batch_projection_bytes, static_cast<size_t>(tokens) * conv_dim * sizeof(float), "batch linear projection");
            ensure_float_buffer(state->batch_z, state->batch_z_bytes, static_cast<size_t>(tokens) * value_total * sizeof(float), "batch linear z");
            ensure_float_buffer(state->batch_b, state->batch_b_bytes, static_cast<size_t>(tokens) * value_heads * sizeof(float), "batch linear b");
            ensure_float_buffer(state->batch_a, state->batch_a_bytes, static_cast<size_t>(tokens) * value_heads * sizeof(float), "batch linear a");
            ensure_float_buffer(state->batch_conv_out, state->batch_conv_out_bytes, static_cast<size_t>(tokens) * conv_dim * sizeof(float), "batch linear conv out");
            ensure_float_buffer(state->batch_gated, state->batch_gated_bytes, static_cast<size_t>(tokens) * value_total * sizeof(float), "batch linear gated");
            ensure_u16_buffer(state->batch_gated_bf16, state->batch_gated_bf16_bytes, static_cast<size_t>(tokens) * value_total * sizeof(uint16_t), "batch linear gated bf16");

            const TensorRef qkv_w = tensor(prefix + "linear_attn.in_proj_qkv.weight");
            const TensorRef z_w = tensor(prefix + "linear_attn.in_proj_z.weight");
            const TensorRef b_w = tensor(prefix + "linear_attn.in_proj_b.weight");
            const TensorRef a_w = tensor(prefix + "linear_attn.in_proj_a.weight");
            const TensorRef conv_w = tensor(prefix + "linear_attn.conv1d.weight");
            const TensorRef a_log = tensor(prefix + "linear_attn.A_log");
            const TensorRef dt_bias = tensor(prefix + "linear_attn.dt_bias");
            const TensorRef attn_norm_w = tensor(prefix + "linear_attn.norm.weight");
            const TensorRef attn_out_w = tensor(prefix + "linear_attn.out_proj.weight");
            DeviceWeight * qkv_device = cached_cuda_weight(qkv_w);
            if (!qkv_device) {
                return nullptr;
            }
            cublas_batch_matvec_to_device(cache, qkv_w, *qkv_device, cache.norm_bf16_buffer, CUDA_R_16BF, tokens, state->batch_projection);
            DeviceWeight * z_device = cached_cuda_weight(z_w);
            if (!z_device) {
                return nullptr;
            }
            cublas_batch_matvec_to_device(cache, z_w, *z_device, cache.norm_bf16_buffer, CUDA_R_16BF, tokens, state->batch_z);
            DeviceWeight * b_device = cached_cuda_weight(b_w);
            if (!b_device) {
                return nullptr;
            }
            cublas_batch_matvec_to_device(cache, b_w, *b_device, cache.norm_bf16_buffer, CUDA_R_16BF, tokens, state->batch_b);
            DeviceWeight * a_device = cached_cuda_weight(a_w);
            if (!a_device) {
                return nullptr;
            }
            cublas_batch_matvec_to_device(cache, a_w, *a_device, cache.norm_bf16_buffer, CUDA_R_16BF, tokens, state->batch_a);
            DeviceWeight * conv_device = cached_cuda_weight(conv_w);
            if (!conv_device) {
                return nullptr;
            }
            launch_linear_attention_conv_batch(
                state->batch_projection,
                static_cast<const uint16_t *>(conv_device->ptr),
                state->conv_state,
                state->batch_conv_out,
                tokens,
                conv_dim,
                config.text.linear_conv_kernel_dim,
                nullptr);
            check_cuda(cudaGetLastError(), "launch_linear_attention_conv_batch 失败");
            DeviceWeight * a_log_device = cached_cuda_weight(a_log);
            DeviceWeight * dt_bias_device = cached_cuda_weight(dt_bias);
            DeviceWeight * attn_norm_device = cached_cuda_weight(attn_norm_w);
            if (!a_log_device || !dt_bias_device || !attn_norm_device) {
                return nullptr;
            }
            launch_linear_attention_recurrent_batch(
                state->batch_conv_out,
                state->batch_z,
                state->batch_b,
                state->batch_a,
                static_cast<const float *>(a_log_device->ptr),
                static_cast<const uint16_t *>(dt_bias_device->ptr),
                static_cast<const float *>(attn_norm_device->ptr),
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
            DeviceWeight * attn_out_device = cached_cuda_weight(attn_out_w);
            if (!attn_out_device) {
                return nullptr;
            }
            launch_float_to_lowp(state->batch_gated, state->batch_gated_bf16, tokens * value_total, attn_out_device->type);
            check_cuda(cudaGetLastError(), "launch_float_to_lowp batch linear gated 失败");
            cublas_batch_matvec_to_device(cache, attn_out_w, *attn_out_device, state->batch_gated_bf16, attn_out_device->type, tokens, cache.mixer_buffer);
        } else {
            const int n_heads = config.text.num_attention_heads;
            const int kv_heads = config.text.num_key_value_heads;
            const int head_dim = config.text.head_dim;
            const int q_total = n_heads * head_dim;
            const int kv_total = kv_heads * head_dim;
            CudaFullAttentionState * state =
                ensure_full_attention_state(full_states[layer], n_heads, kv_heads, head_dim, full_max_seq_lens[layer]);
            ensure_float_buffer(state->batch_projection, state->batch_projection_bytes, static_cast<size_t>(tokens) * q_total * 2 * sizeof(float), "batch full q projection");
            ensure_float_buffer(state->batch_k, state->batch_k_bytes, static_cast<size_t>(tokens) * kv_total * sizeof(float), "batch full k");
            ensure_float_buffer(state->batch_v, state->batch_v_bytes, static_cast<size_t>(tokens) * kv_total * sizeof(float), "batch full v");
            ensure_float_buffer(state->batch_q, state->batch_q_bytes, static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full q");
            ensure_float_buffer(state->batch_gate, state->batch_gate_bytes, static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full gate");
            ensure_float_buffer(state->batch_attn, state->batch_attn_bytes, static_cast<size_t>(tokens) * q_total * sizeof(float), "batch full attn");
            ensure_u16_buffer(state->batch_attn_bf16, state->batch_attn_bf16_bytes, static_cast<size_t>(tokens) * q_total * sizeof(uint16_t), "batch full attn bf16");

            const TensorRef q_w = tensor(prefix + "self_attn.q_proj.weight");
            const TensorRef k_w = tensor(prefix + "self_attn.k_proj.weight");
            const TensorRef v_w = tensor(prefix + "self_attn.v_proj.weight");
            const TensorRef q_norm_w = tensor(prefix + "self_attn.q_norm.weight");
            const TensorRef k_norm_w = tensor(prefix + "self_attn.k_norm.weight");
            const TensorRef out_w = tensor(prefix + "self_attn.o_proj.weight");
            DeviceWeight * q_device = cached_cuda_weight(q_w);
            if (!q_device) {
                return nullptr;
            }
            cublas_batch_matvec_to_device(cache, q_w, *q_device, cache.norm_bf16_buffer, CUDA_R_16BF, tokens, state->batch_projection);
            DeviceWeight * k_device = cached_cuda_weight(k_w);
            if (!k_device) {
                return nullptr;
            }
            cublas_batch_matvec_to_device(cache, k_w, *k_device, cache.norm_bf16_buffer, CUDA_R_16BF, tokens, state->batch_k);
            DeviceWeight * v_device = cached_cuda_weight(v_w);
            if (!v_device) {
                return nullptr;
            }
            cublas_batch_matvec_to_device(cache, v_w, *v_device, cache.norm_bf16_buffer, CUDA_R_16BF, tokens, state->batch_v);
            DeviceWeight * q_norm_device = cached_cuda_weight(q_norm_w);
            if (!q_norm_device) {
                return nullptr;
            }
            launch_full_attention_q_batch(
                state->batch_projection,
                static_cast<const uint16_t *>(q_norm_device->ptr),
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
            DeviceWeight * k_norm_device = cached_cuda_weight(k_norm_w);
            if (!k_norm_device) {
                return nullptr;
            }
            launch_full_attention_kv_batch(
                state->batch_k,
                state->batch_v,
                static_cast<const uint16_t *>(k_norm_device->ptr),
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
            DeviceWeight * out_device = cached_cuda_weight(out_w);
            if (!out_device) {
                return nullptr;
            }
            launch_float_to_lowp(state->batch_attn, state->batch_attn_bf16, tokens * q_total, out_device->type);
            check_cuda(cudaGetLastError(), "launch_float_to_lowp batch full attn 失败");
            cublas_batch_matvec_to_device(cache, out_w, *out_device, state->batch_attn_bf16, out_device->type, tokens, cache.mixer_buffer);
        }

        const TensorRef post_norm_w = tensor(prefix + "post_attention_layernorm.weight");
        DeviceWeight * post_norm_device = cached_cuda_weight(post_norm_w);
        if (!post_norm_device) {
            return nullptr;
        }
        launch_add_rms_norm_batch_to_bf16(
            current,
            cache.mixer_buffer,
            static_cast<const uint16_t *>(post_norm_device->ptr),
            cache.layer_out_buffer,
            cache.post_norm_bf16_buffer,
            tokens,
            hidden_dim,
            config.text.rms_norm_eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_add_rms_norm_batch_to_bf16 post 失败");
        if (!cuda_mlp_batch_from_device_bf16_to_device(
                tensor(prefix + "mlp.gate_proj.weight"),
                tensor(prefix + "mlp.up_proj.weight"),
                tensor(prefix + "mlp.down_proj.weight"),
                cache.post_norm_bf16_buffer,
                tokens,
                cache.mlp_out_buffer)) {
            return nullptr;
        }
        launch_add_float_batch(cache.layer_out_buffer, cache.mlp_out_buffer, next, tokens * hidden_dim, nullptr);
        check_cuda(cudaGetLastError(), "launch_add_float_batch mlp residual 失败");
        std::swap(current, next);
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize batch prefill 失败");
    for (void * state_handle : linear_states) {
        release_linear_attention_batch_buffers(static_cast<CudaLinearAttentionState *>(state_handle));
    }
    for (void * state_handle : full_states) {
        release_full_attention_batch_buffers(static_cast<CudaFullAttentionState *>(state_handle));
    }
    seq_len += tokens;
    return current + static_cast<size_t>(tokens - 1) * hidden_dim;
#else
    (void) config;
    (void) weights;
    (void) prompt_ids;
    (void) linear_states;
    (void) full_states;
    (void) full_max_seq_lens;
    (void) seq_len;
    return nullptr;
#endif
}

bool cuda_rmsnorm_mlp_layer(
    const TensorRef & norm_w,
    const TensorRef & gate_w,
    const TensorRef & up_w,
    const TensorRef & down_w,
    const std::vector<float> & x,
    float eps,
    bool one_plus,
    std::vector<float> & out) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (!cuda_rmsnorm_mlp_enabled()) {
        return false;
    }
    if (norm_w.info->shape.size() != 1 ||
        gate_w.info->shape.size() != 2 ||
        up_w.info->shape.size() != 2 ||
        down_w.info->shape.size() != 2) {
        return false;
    }
    if (norm_w.info->dtype != "BF16" ||
        gate_w.info->dtype != "BF16" ||
        up_w.info->dtype != "BF16" ||
        down_w.info->dtype != "BF16") {
        return false;
    }

    const int hidden_dim = static_cast<int>(norm_w.info->shape[0]);
    const int intermediate_dim = static_cast<int>(gate_w.info->shape[0]);
    if (gate_w.info->shape[1] != hidden_dim ||
        up_w.info->shape[0] != intermediate_dim ||
        up_w.info->shape[1] != hidden_dim ||
        down_w.info->shape[0] != hidden_dim ||
        down_w.info->shape[1] != intermediate_dim ||
        static_cast<int>(x.size()) != hidden_dim) {
        return false;
    }

    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    if (!norm_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t hidden_float_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t hidden_bf16_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_float_buffer(cache.norm_input_buffer, cache.norm_input_bytes, hidden_float_bytes, "norm input buffer");
    ensure_u16_buffer(cache.norm_bf16_buffer, cache.norm_bf16_bytes, hidden_bf16_bytes, "norm bf16 buffer");
    check_cuda(cudaMemcpy(cache.norm_input_buffer, x.data(), hidden_float_bytes, cudaMemcpyHostToDevice), "cudaMemcpy norm input 失败");
    launch_rms_norm_to_bf16(
        cache.norm_input_buffer,
        static_cast<const uint16_t *>(norm_device->ptr),
        cache.norm_bf16_buffer,
        hidden_dim,
        eps,
        one_plus,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 失败");
    return cuda_mlp_from_device_bf16(gate_w, up_w, down_w, cache.norm_bf16_buffer, out);
#else
    (void) norm_w;
    (void) gate_w;
    (void) up_w;
    (void) down_w;
    (void) x;
    (void) eps;
    (void) one_plus;
    (void) out;
    return false;
#endif
}

bool cuda_cublas_enabled() {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    return true;
#else
    return false;
#endif
}

void cuda_free_linear_attention_state(void * state) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    delete static_cast<CudaLinearAttentionState *>(state);
#else
    (void) state;
#endif
}

void cuda_free_full_attention_state(void * state) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    delete static_cast<CudaFullAttentionState *>(state);
#else
    (void) state;
#endif
}

bool cuda_fused_mlp_enabled() {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    return true;
#else
    return false;
#endif
}

bool cuda_project_attention_enabled() {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    return true;
#else
    return false;
#endif
}

bool cuda_full_layer_enabled() {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    return true;
#else
    return false;
#endif
}

bool cuda_rmsnorm_mlp_enabled() {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    static const bool enabled = [] {
        const char * value = std::getenv("LLM_INFERENCE_CUDA_FUSE_RMSNORM_MLP");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
#else
    return false;
#endif
}

} // namespace llm_inference
