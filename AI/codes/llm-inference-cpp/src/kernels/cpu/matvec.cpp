#include "cpu_ops.h"

#include <stdexcept>

namespace llm_inference {
namespace cpu {

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

} // namespace cpu
} // namespace llm_inference
