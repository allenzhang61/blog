#include "cpu_ops.h"

#include <cmath>

namespace llm_inference {
namespace cpu {

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

} // namespace cpu
} // namespace llm_inference
