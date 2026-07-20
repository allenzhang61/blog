#include "llm/model/Linear.hpp"

#include "llm/ops.hpp"

namespace llm {

Linear::Linear(int64_t in_features, int64_t out_features, bool bias_enabled, Device device)
    : weight(Tensor::randn({in_features, out_features}, 0.02, device, true)),
      bias(Tensor::zeros({out_features}, device, true)),
      use_bias(bias_enabled) {}

Tensor Linear::forward(const Tensor& x) {
    int64_t in_features = weight.shape()[0];
    int64_t out_features = weight.shape()[1];
    std::vector<int64_t> out_shape = x.shape();
    if (out_shape.empty() || out_shape.back() != in_features) {
        throw std::runtime_error("Linear input shape mismatch");
    }
    int64_t rows = x.numel() / in_features;
    Tensor flat = ops::reshape(x, {rows, in_features});
    Tensor y = ops::matmul(flat, weight);
    if (use_bias) {
        y = ops::add(y, bias);
    }
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
