#include "cpu_ops.h"

#include <cmath>

namespace llm_inference {
namespace cpu {

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

void add_inplace(std::vector<float> & x, const std::vector<float> & y) {
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += y[i];
    }
}

} // namespace cpu
} // namespace llm_inference
