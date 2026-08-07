#include "cuda_weight_cache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "cuda_common.h"

namespace llm_inference {

namespace {

size_t cuda_weight_cache_limit_bytes() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_WEIGHT_CACHE_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 10.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

template <typename T>
void cuda_free_if_set(T * ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

} // namespace

DeviceWeight::~DeviceWeight() {
    cuda_free_if_set(ptr);
}

DeviceWeight::DeviceWeight(DeviceWeight && other) noexcept {
    ptr = other.ptr;
    bytes = other.bytes;
    type = other.type;
    other.ptr = nullptr;
    other.bytes = 0;
}

DeviceWeight & DeviceWeight::operator=(DeviceWeight && other) noexcept {
    if (this != &other) {
        cuda_free_if_set(ptr);
        ptr = other.ptr;
        bytes = other.bytes;
        type = other.type;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    return *this;
}

CudaWeightCache::CudaWeightCache() {
    check_cublas(cublasCreate(&handle), "cublasCreate 失败");
}

CudaWeightCache::~CudaWeightCache() {
    if (handle) {
        cublasDestroy(handle);
    }
}

CudaLinearAttentionState::~CudaLinearAttentionState() {
    cuda_free_if_set(conv_state);
    cuda_free_if_set(recurrent_state);
    cuda_free_if_set(mixed);
    cuda_free_if_set(projection);
    cuda_free_if_set(z);
    cuda_free_if_set(b);
    cuda_free_if_set(a);
    cuda_free_if_set(conv_out);
    cuda_free_if_set(gated);
    cuda_free_if_set(gated_bf16);
}

CudaFullAttentionState::~CudaFullAttentionState() {
    cuda_free_if_set(q_and_gate);
    cuda_free_if_set(projection);
    cuda_free_if_set(k);
    cuda_free_if_set(v);
    cuda_free_if_set(q);
    cuda_free_if_set(gate);
    cuda_free_if_set(key_cache);
    cuda_free_if_set(value_cache);
    cuda_free_if_set(attn);
    cuda_free_if_set(attn_bf16);
}

CudaWeightCache & cuda_weight_cache() {
    static CudaWeightCache cache;
    return cache;
}

cudaDataType_t cuda_type_for(const WeightData & weight) {
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

size_t dtype_size_for(const WeightData & weight) {
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

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

DeviceWeight * cached_cuda_weight(const WeightData & weight) {
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

DeviceWeight * cached_cuda_concat_weight(const std::string & name, const std::vector<WeightData> & weights) {
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
    for (const WeightData & weight : weights) {
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

    std::vector<uint16_t> host;
    host.reserve(elems);
    for (const WeightData & weight : weights) {
        const size_t weight_elems = static_cast<size_t>(weight.info->shape[0]) * static_cast<size_t>(in_dim);
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data);
        host.insert(host.end(), p, p + weight_elems);
    }

    DeviceWeight device;
    device.bytes = bytes;
    device.type = CUDA_R_16BF;
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc concat weight 失败 " + name);
    check_cuda(cudaMemcpy(device.ptr, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy concat weight 失败 " + name);
    auto [it, inserted] = cache.items.emplace(name, std::move(device));
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

} // namespace llm_inference
