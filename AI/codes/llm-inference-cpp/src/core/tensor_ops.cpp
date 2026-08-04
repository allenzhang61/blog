#include "llm_inference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#ifdef LLM_INFERENCE_USE_CBLAS
#include <cblas.h>
#endif

#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
#include "cuda_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace llm_inference {

float bf16_to_float(uint16_t value);
float f16_to_float(uint16_t h);

namespace {

#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS

size_t cuda_weight_cache_limit_bytes() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_WEIGHT_CACHE_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 10.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
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
    float * norm_input_buffer = nullptr;
    size_t norm_input_bytes = 0;
    uint16_t * norm_bf16_buffer = nullptr;
    size_t norm_bf16_bytes = 0;
    float * argmax_block_values = nullptr;
    size_t argmax_block_values_bytes = 0;
    int * argmax_block_indices = nullptr;
    size_t argmax_block_indices_bytes = 0;
    float * argmax_best_value = nullptr;
    int * argmax_best_index = nullptr;
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
        if (norm_input_buffer) {
            cudaFree(norm_input_buffer);
        }
        if (norm_bf16_buffer) {
            cudaFree(norm_bf16_buffer);
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
    float * z = nullptr;
    float * b = nullptr;
    float * a = nullptr;
    float * conv_out = nullptr;
    float * gated = nullptr;
    uint16_t * gated_bf16 = nullptr;

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
    }
};

struct CudaFullAttentionState {
    int n_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int max_seq_len = 0;
    float * q_and_gate = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    float * q = nullptr;
    float * gate = nullptr;
    float * key_cache = nullptr;
    float * value_cache = nullptr;
    float * attn = nullptr;
    uint16_t * attn_bf16 = nullptr;

    ~CudaFullAttentionState() {
        if (q_and_gate) {
            cudaFree(q_and_gate);
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
    check_cuda(cudaMemcpy(device.ptr, weight.data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy weight 失败 " + weight.info->name);
    auto [it, inserted] = cache.items.emplace(weight.info->name, std::move(device));
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

void cublas_matvec_to_device(
    CudaWeightCache & cache,
    const TensorRef & weight,
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

    DeviceWeight * gate_device = cached_cuda_weight(gate_w);
    DeviceWeight * up_device = cached_cuda_weight(up_w);
    DeviceWeight * down_device = cached_cuda_weight(down_w);
    if (!gate_device || !up_device || !down_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const size_t intermediate_float_bytes = static_cast<size_t>(intermediate_dim) * sizeof(float);
    const size_t intermediate_bf16_bytes = static_cast<size_t>(intermediate_dim) * sizeof(uint16_t);
    const size_t hidden_float_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    ensure_float_buffer(cache.gate_buffer, cache.gate_bytes, intermediate_float_bytes, "mlp gate buffer");
    ensure_float_buffer(cache.up_buffer, cache.up_bytes, intermediate_float_bytes, "mlp up buffer");
    ensure_float_buffer(cache.prod_buffer, cache.prod_bytes, intermediate_float_bytes, "mlp prod buffer");
    ensure_u16_buffer(cache.prod_bf16_buffer, cache.prod_bf16_bytes, intermediate_bf16_bytes, "mlp prod bf16 buffer");
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, hidden_float_bytes, "mlp out buffer");

    cublas_matvec_to_device(cache, gate_w, *gate_device, device_x, CUDA_R_16BF, cache.gate_buffer);
    cublas_matvec_to_device(cache, up_w, *up_device, device_x, CUDA_R_16BF, cache.up_buffer);
    launch_silu_mul(cache.gate_buffer, cache.up_buffer, cache.prod_buffer, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul 失败");
    launch_float_to_bf16(cache.prod_buffer, cache.prod_bf16_buffer, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 失败");
    cublas_matvec_to_device(cache, down_w, *down_device, cache.prod_bf16_buffer, CUDA_R_16BF, cache.out_buffer);

    out.assign(static_cast<size_t>(hidden_dim), 0.0f);
    check_cuda(cudaMemcpy(out.data(), cache.out_buffer, hidden_float_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy mlp out 失败");
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

bool cuda_matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y) {
    DeviceWeight * device_weight = cached_cuda_weight(weight);
    if (!device_weight) {
        return false;
    }
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    y.assign(out_dim, 0.0f);

    auto & cache = cuda_weight_cache();
    std::vector<uint16_t> x_bf16;
    cudaDataType_t x_type = CUDA_R_32F;
    size_t x_bytes = static_cast<size_t>(in_dim) * sizeof(float);
    if (device_weight->type == CUDA_R_16BF) {
        x_type = CUDA_R_16BF;
        x_bytes = static_cast<size_t>(in_dim) * sizeof(uint16_t);
        x_bf16 = host_float_to_bf16(x);
    }

    const size_t y_bytes = static_cast<size_t>(out_dim) * sizeof(float);
    ensure_cuda_buffers(cache, x_bytes, y_bytes);
    check_cuda(
        cudaMemcpy(
            cache.x_buffer,
            x_type == CUDA_R_16BF ? static_cast<const void *>(x_bf16.data()) : static_cast<const void *>(x.data()),
            x_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy x 失败");

    cublas_matvec_to_device(cache, weight, *device_weight, cache.x_buffer, x_type, cache.y_buffer);
    check_cuda(cudaMemcpy(y.data(), cache.y_buffer, y_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy y 失败");
    return true;
}

#endif

#ifdef LLM_INFERENCE_USE_CBLAS

size_t weight_cache_limit_bytes() {
    const char * env = std::getenv("LLM_INFERENCE_WEIGHT_CACHE_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 8.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

struct FloatWeightCache {
    std::mutex mutex;
    size_t bytes = 0;
    std::unordered_map<std::string, std::vector<float>> items;
};

FloatWeightCache & float_weight_cache() {
    static FloatWeightCache cache;
    return cache;
}

const std::vector<float> * cached_float_weight(const TensorRef & weight) {
    const size_t elems = static_cast<size_t>(weight.info->shape[0]) * static_cast<size_t>(weight.info->shape[1]);
    const size_t bytes = elems * sizeof(float);
    const size_t limit = weight_cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }

    auto & cache = float_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto found = cache.items.find(weight.info->name);
    if (found != cache.items.end()) {
        return &found->second;
    }
    if (cache.bytes + bytes > limit) {
        cache.items.clear();
        cache.bytes = 0;
    }

    auto [it, inserted] = cache.items.emplace(weight.info->name, std::vector<float>());
    std::vector<float> & data = it->second;
    data.resize(elems);
    if (weight.info->dtype == "BF16") {
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data);
        for (size_t i = 0; i < elems; ++i) {
            data[i] = bf16_to_float(p[i]);
        }
    } else if (weight.info->dtype == "F16") {
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data);
        for (size_t i = 0; i < elems; ++i) {
            data[i] = f16_to_float(p[i]);
        }
    } else if (weight.info->dtype == "F32") {
        const auto * p = reinterpret_cast<const float *>(weight.data);
        std::copy(p, p + elems, data.begin());
    } else {
        throw std::runtime_error("暂不支持 dtype：" + weight.info->dtype + " tensor=" + weight.info->name);
    }
    cache.bytes += bytes;
    (void) inserted;
    return &data;
}

#endif

} // namespace

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float f16_to_float(uint16_t h) {
    const uint16_t h_exp = h & 0x7C00u;
    const uint16_t h_sig = h & 0x03FFu;
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t f;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f = sign;
        } else {
            int exp = -1;
            uint16_t sig = h_sig;
            do {
                ++exp;
                sig <<= 1;
            } while ((sig & 0x0400u) == 0);
            sig &= 0x03FFu;
            f = sign | static_cast<uint32_t>(127 - 15 - exp) << 23 | static_cast<uint32_t>(sig) << 13;
        }
    } else if (h_exp == 0x7C00u) {
        f = sign | 0x7F800000u | (static_cast<uint32_t>(h_sig) << 13);
    } else {
        f = sign | (static_cast<uint32_t>((h_exp >> 10) + (127 - 15)) << 23) | (static_cast<uint32_t>(h_sig) << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

float tensor_value(const TensorRef & ref, size_t index) {
    if (ref.info->dtype == "BF16") {
        const auto * p = reinterpret_cast<const uint16_t *>(ref.data);
        return bf16_to_float(p[index]);
    }
    if (ref.info->dtype == "F16") {
        const auto * p = reinterpret_cast<const uint16_t *>(ref.data);
        return f16_to_float(p[index]);
    }
    if (ref.info->dtype == "F32") {
        const auto * p = reinterpret_cast<const float *>(ref.data);
        return p[index];
    }
    throw std::runtime_error("暂不支持 dtype：" + ref.info->dtype + " tensor=" + ref.info->name);
}

float dot_row(const TensorRef & weight, int row, const std::vector<float> & x) {
    if (weight.info->shape.size() != 2) {
        throw std::runtime_error("dot_row 需要二维权重：" + weight.info->name);
    }
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    if (row < 0 || row >= out_dim || static_cast<int>(x.size()) != in_dim) {
        throw std::runtime_error("dot_row 维度不匹配：" + weight.info->name);
    }
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(in_dim);
    float sum = 0.0f;

    if (weight.info->dtype == "BF16") {
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data) + base;
        for (int i = 0; i < in_dim; ++i) {
            sum += bf16_to_float(p[i]) * x[i];
        }
        return sum;
    }
    if (weight.info->dtype == "F16") {
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data) + base;
        for (int i = 0; i < in_dim; ++i) {
            sum += f16_to_float(p[i]) * x[i];
        }
        return sum;
    }
    if (weight.info->dtype == "F32") {
        const auto * p = reinterpret_cast<const float *>(weight.data) + base;
        for (int i = 0; i < in_dim; ++i) {
            sum += p[i] * x[i];
        }
        return sum;
    }
    throw std::runtime_error("暂不支持 dtype：" + weight.info->dtype + " tensor=" + weight.info->name);
}

void matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y) {
    if (weight.info->shape.size() != 2) {
        throw std::runtime_error("matvec 需要二维权重：" + weight.info->name);
    }
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    if (static_cast<int>(x.size()) != in_dim) {
        throw std::runtime_error("matvec 输入维度不匹配：" + weight.info->name);
    }
    y.assign(out_dim, 0.0f);

#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    if (cuda_matvec(weight, x, y)) {
        return;
    }
#endif

#ifdef LLM_INFERENCE_USE_CBLAS
    if (const std::vector<float> * matrix = cached_float_weight(weight)) {
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            out_dim,
            in_dim,
            1.0f,
            matrix->data(),
            in_dim,
            x.data(),
            1,
            0.0f,
            y.data(),
            1);
        return;
    }
#endif

#pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; ++o) {
        y[o] = dot_row(weight, o, x);
    }
}

bool cuda_argmax_matvec(const TensorRef & weight, const std::vector<float> & x, int & best_id) {
#ifdef LLM_INFERENCE_USE_CUDA_CUBLAS
    DeviceWeight * device_weight = cached_cuda_weight(weight);
    if (!device_weight || device_weight->type != CUDA_R_16BF) {
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
    std::vector<uint16_t> x_bf16 = host_float_to_bf16(x);
    check_cuda(cudaMemcpy(cache.x_buffer, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy argmax x 失败");
    cublas_matvec_to_device(cache, weight, *device_weight, cache.x_buffer, CUDA_R_16BF, cache.y_buffer);

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

void embedding_lookup(const TensorRef & emb, int token_id, std::vector<float> & y) {
    const int vocab = static_cast<int>(emb.info->shape[0]);
    const int hidden = static_cast<int>(emb.info->shape[1]);
    if (token_id < 0 || token_id >= vocab) {
        throw std::runtime_error("token id 越界：" + std::to_string(token_id));
    }
    y.resize(hidden);
    const size_t base = static_cast<size_t>(token_id) * static_cast<size_t>(hidden);
    for (int i = 0; i < hidden; ++i) {
        y[i] = tensor_value(emb, base + static_cast<size_t>(i));
    }
}

void rms_norm(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y, float eps, bool one_plus) {
    const int dim = static_cast<int>(x.size());
    double ss = 0.0;
    for (float v : x) {
        ss += static_cast<double>(v) * v;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    y.resize(dim);
    for (int i = 0; i < dim; ++i) {
        const float w = tensor_value(weight, static_cast<size_t>(i));
        y[i] = x[i] * scale * (one_plus ? (1.0f + w) : w);
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float silu(float x) {
    return x * sigmoid(x);
}

float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

void l2_norm_inplace(float * x, int dim, float eps) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) {
        ss += static_cast<double>(x[i]) * x[i];
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss) + eps);
    for (int i = 0; i < dim; ++i) {
        x[i] *= scale;
    }
}

void gated_rms_norm_head(
    const TensorRef & weight,
    const float * x,
    const float * gate,
    float * y,
    int dim,
    float eps) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) {
        ss += static_cast<double>(x[i]) * x[i];
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    for (int i = 0; i < dim; ++i) {
        y[i] = tensor_value(weight, static_cast<size_t>(i)) * x[i] * scale * silu(gate[i]);
    }
}

void add_inplace(std::vector<float> & x, const std::vector<float> & y) {
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += y[i];
    }
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
    const size_t x_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_cuda_buffers(cache, x_bytes, 1);

    std::vector<uint16_t> x_bf16 = host_float_to_bf16(x);
    check_cuda(cudaMemcpy(cache.x_buffer, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy mlp x 失败");
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

    DeviceWeight * qkv_device = cached_cuda_weight(in_proj_qkv_w);
    DeviceWeight * z_device = cached_cuda_weight(in_proj_z_w);
    DeviceWeight * b_device = cached_cuda_weight(in_proj_b_w);
    DeviceWeight * a_device = cached_cuda_weight(in_proj_a_w);
    DeviceWeight * conv_device = cached_cuda_weight(conv_w);
    DeviceWeight * a_log_device = cached_cuda_weight(a_log);
    DeviceWeight * dt_bias_device = cached_cuda_weight(dt_bias);
    DeviceWeight * norm_device = cached_cuda_weight(norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!qkv_device || !z_device || !b_device || !a_device || !conv_device || !a_log_device || !dt_bias_device || !norm_device || !out_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const int hidden_dim = static_cast<int>(x.size());
    const size_t x_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_cuda_buffers(cache, x_bytes, 1);
    std::vector<uint16_t> x_bf16 = host_float_to_bf16(x);
    check_cuda(cudaMemcpy(cache.x_buffer, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy linear project x 失败");

    CudaLinearAttentionState * state =
        ensure_linear_attention_state(state_handle, key_heads, value_heads, k_dim, v_dim, kernel);
    cublas_matvec_to_device(cache, in_proj_qkv_w, *qkv_device, cache.x_buffer, CUDA_R_16BF, state->mixed);
    cublas_matvec_to_device(cache, in_proj_z_w, *z_device, cache.x_buffer, CUDA_R_16BF, state->z);
    cublas_matvec_to_device(cache, in_proj_b_w, *b_device, cache.x_buffer, CUDA_R_16BF, state->b);
    cublas_matvec_to_device(cache, in_proj_a_w, *a_device, cache.x_buffer, CUDA_R_16BF, state->a);

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

    const int out_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(out_dim) * sizeof(float), "linear out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->gated_bf16, CUDA_R_16BF, cache.out_buffer);
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

    DeviceWeight * q_proj_device = cached_cuda_weight(q_proj_w);
    DeviceWeight * k_proj_device = cached_cuda_weight(k_proj_w);
    DeviceWeight * v_proj_device = cached_cuda_weight(v_proj_w);
    DeviceWeight * q_norm_device = cached_cuda_weight(q_norm_w);
    DeviceWeight * k_norm_device = cached_cuda_weight(k_norm_w);
    DeviceWeight * out_device = cached_cuda_weight(out_w);
    if (!q_proj_device || !k_proj_device || !v_proj_device || !q_norm_device || !k_norm_device || !out_device) {
        return false;
    }

    auto & cache = cuda_weight_cache();
    const int hidden_dim = static_cast<int>(x.size());
    const size_t x_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    ensure_cuda_buffers(cache, x_bytes, 1);
    std::vector<uint16_t> x_bf16 = host_float_to_bf16(x);
    check_cuda(cudaMemcpy(cache.x_buffer, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy full project x 失败");

    CudaFullAttentionState * state =
        ensure_full_attention_state(state_handle, n_heads, kv_heads, head_dim, max_seq_len);
    cublas_matvec_to_device(cache, q_proj_w, *q_proj_device, cache.x_buffer, CUDA_R_16BF, state->q_and_gate);
    cublas_matvec_to_device(cache, k_proj_w, *k_proj_device, cache.x_buffer, CUDA_R_16BF, state->k);
    cublas_matvec_to_device(cache, v_proj_w, *v_proj_device, cache.x_buffer, CUDA_R_16BF, state->v);

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

    const int out_dim = static_cast<int>(out_w.info->shape[0]);
    ensure_float_buffer(cache.out_buffer, cache.out_bytes, static_cast<size_t>(out_dim) * sizeof(float), "full out buffer");
    cublas_matvec_to_device(cache, out_w, *out_device, state->attn_bf16, CUDA_R_16BF, cache.out_buffer);
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
