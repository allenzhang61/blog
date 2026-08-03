#include "llm_inference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llm_inference {

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

#pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; ++o) {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(o) * static_cast<size_t>(in_dim);
        for (int i = 0; i < in_dim; ++i) {
            sum += static_cast<double>(tensor_value(weight, base + static_cast<size_t>(i))) * x[i];
        }
        y[o] = static_cast<float>(sum);
    }
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

} // namespace llm_inference
