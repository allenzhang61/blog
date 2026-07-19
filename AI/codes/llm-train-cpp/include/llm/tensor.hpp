#pragma once

#include "llm/core.hpp"

namespace llm {

struct TensorNode;

class Tensor {
public:
    std::shared_ptr<TensorNode> node;

    Tensor();
    Tensor(std::vector<int64_t> shape, DType dtype = DType::Float32,
           Device device = {}, bool requires_grad = false);
    Tensor(std::vector<int64_t> shape, std::vector<double> data,
           DType dtype = DType::Float32, Device device = {}, bool requires_grad = false);

    static Tensor zeros(const std::vector<int64_t>& shape, Device device = {}, bool requires_grad = false);
    static Tensor ones(const std::vector<int64_t>& shape, Device device = {}, bool requires_grad = false);
    static Tensor randn(const std::vector<int64_t>& shape, double scale = 0.02,
                        Device device = {}, bool requires_grad = false);
    static Tensor from_vector(const std::vector<double>& values, const std::vector<int64_t>& shape,
                              Device device = {}, bool requires_grad = false);
    static Tensor from_ints(const std::vector<int64_t>& values, const std::vector<int64_t>& shape,
                            Device device = {});

    const std::vector<int64_t>& shape() const;
    DType dtype() const;
    Device device() const;
    bool requires_grad() const;
    int64_t numel() const;
    std::vector<double>& data();
    const std::vector<double>& data() const;
    std::vector<double>& grad() const;
    double item() const;
    void zero_grad();
    void backward();
};

struct TensorNode {
    std::vector<int64_t> shape;
    DType dtype{DType::Float32};
    Device device{};
    std::vector<double> data;
    std::vector<double> grad;
    bool requires_grad{false};
    std::vector<Tensor> parents;
    std::function<void()> backward_fn;
};

void ensure_cpu(const Tensor& t);
std::vector<int64_t> strides_for(const std::vector<int64_t>& shape);

} // namespace llm
