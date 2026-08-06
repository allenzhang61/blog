#include "tensor_ops.h"

#include "../kernels/cpu/cpu_ops.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace llm_inference {
namespace ops {

namespace {

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
            data[i] = cpu::bf16_to_float(p[i]);
        }
    } else if (weight.info->dtype == "F16") {
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data);
        for (size_t i = 0; i < elems; ++i) {
            data[i] = cpu::f16_to_float(p[i]);
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

} // namespace

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

#pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; ++o) {
        y[o] = cpu::dot_row(weight, o, x);
    }
}

} // namespace ops
} // namespace llm_inference
