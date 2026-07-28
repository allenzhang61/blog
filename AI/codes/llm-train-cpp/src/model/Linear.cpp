#include "llm/model/Linear.hpp"

#include "llm/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace llm {

// 权重初始化对齐 PyTorch nn.Linear 的默认（Kaiming uniform）：
//   weight, bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))，fan_in = in_features。
// 固定 std=0.02 在浅层可用，但深层（如 12 层 GPT）残差流方差累积会导致训练停滞，
// 因此改用与 PyTorch 一致的按 fan_in 缩放的均匀分布初始化。
Linear::Linear(int64_t in_features, int64_t out_features, bool bias_enabled, Device device)
    : weight(Tensor::uniform({in_features, out_features},
                             1.0 / std::sqrt(static_cast<double>(in_features)), device, true)),
      bias(Tensor::uniform({out_features},
                           1.0 / std::sqrt(static_cast<double>(in_features)), device, true)),
      use_bias(bias_enabled) {}

// x: (..., in_features) -> 返回: (..., out_features)
// weight: (in_features, out_features), bias: (out_features)
Tensor Linear::forward(const Tensor& x) {
    int64_t in_features = weight.shape()[0];
    int64_t out_features = weight.shape()[1];
    std::vector<int64_t> out_shape = x.shape();
    if (out_shape.empty() || out_shape.back() != in_features) {
        throw std::runtime_error("Linear input shape mismatch");
    }
    // 把前面所有维度压平成一个二维矩阵：(..., in_features) -> (rows, in_features)
    int64_t rows = x.numel() / in_features;
    Tensor flat = ops::reshape(x, {rows, in_features});
    // (rows, in_features) @ (in_features, out_features) -> (rows, out_features)
    Tensor y = ops::matmul(flat, weight);
    if (use_bias) {
        y = ops::add(y, bias);  // bias (out_features) 广播到 (rows, out_features)
    }
    // 还原成输入的前置维度，末维换成 out_features
    out_shape.back() = out_features;
    return ops::reshape(y, out_shape);
}

std::vector<Tensor*> Linear::parameters() {
    if (use_bias) {
        return {&weight, &bias};
    }
    return {&weight};
}

} // namespace llm
