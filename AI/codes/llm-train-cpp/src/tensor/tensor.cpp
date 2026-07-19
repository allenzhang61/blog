#include "llm/tensor.hpp"

namespace llm {

Tensor::Tensor() : node(std::make_shared<TensorNode>()) {}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device, bool requires_grad)
    : node(std::make_shared<TensorNode>()) {
    node->shape = std::move(shape);
    node->dtype = dtype;
    node->device = device;
    node->requires_grad = requires_grad;
    node->data.assign(product(node->shape), 0.0);
    if (requires_grad) node->grad.assign(node->data.size(), 0.0);
}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<double> data, DType dtype,
               Device device, bool requires_grad)
    : node(std::make_shared<TensorNode>()) {
    node->shape = std::move(shape);
    node->dtype = dtype;
    node->device = device;
    node->requires_grad = requires_grad;
    node->data = std::move(data);
    if (static_cast<int64_t>(node->data.size()) != product(node->shape)) {
        throw std::runtime_error("tensor data size does not match shape");
    }
    if (requires_grad) node->grad.assign(node->data.size(), 0.0);
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, Device device, bool requires_grad) {
    return Tensor(shape, DType::Float32, device, requires_grad);
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, Device device, bool requires_grad) {
    Tensor t(shape, DType::Float32, device, requires_grad);
    std::fill(t.data().begin(), t.data().end(), 1.0);
    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, double scale, Device device, bool requires_grad) {
    Tensor t(shape, DType::Float32, device, requires_grad);
    static std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, scale);
    for (auto& v : t.data()) v = dist(gen);
    return t;
}

Tensor Tensor::from_vector(const std::vector<double>& values, const std::vector<int64_t>& shape,
                           Device device, bool requires_grad) {
    return Tensor(shape, values, DType::Float32, device, requires_grad);
}

Tensor Tensor::from_ints(const std::vector<int64_t>& values, const std::vector<int64_t>& shape, Device device) {
    std::vector<double> data(values.begin(), values.end());
    return Tensor(shape, data, DType::Int64, device, false);
}

const std::vector<int64_t>& Tensor::shape() const { return node->shape; }
DType Tensor::dtype() const { return node->dtype; }
Device Tensor::device() const { return node->device; }
bool Tensor::requires_grad() const { return node->requires_grad; }
int64_t Tensor::numel() const { return static_cast<int64_t>(node->data.size()); }
std::vector<double>& Tensor::data() { return node->data; }
const std::vector<double>& Tensor::data() const { return node->data; }
std::vector<double>& Tensor::grad() const {
    if (node->grad.empty()) node->grad.assign(node->data.size(), 0.0);
    return node->grad;
}
double Tensor::item() const {
    if (numel() != 1) throw std::runtime_error("item() requires scalar tensor");
    return node->data[0];
}
void Tensor::zero_grad() {
    node->grad.assign(node->data.size(), 0.0);
}

namespace {
void topo_visit(const Tensor& t, std::unordered_map<TensorNode*, bool>& seen, std::vector<Tensor>& order) {
    TensorNode* ptr = t.node.get();
    if (seen[ptr]) return;
    seen[ptr] = true;
    for (const auto& parent : t.node->parents) topo_visit(parent, seen, order);
    order.push_back(t);
}
} // namespace

void Tensor::backward() {
    if (numel() != 1) throw std::runtime_error("backward() expects a scalar tensor");
    std::unordered_map<TensorNode*, bool> seen;
    std::vector<Tensor> order;
    topo_visit(*this, seen, order);
    grad().assign(1, 1.0);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (it->node->backward_fn) it->node->backward_fn();
    }
}

void ensure_cpu(const Tensor& t) {
    if (t.device().type != DeviceType::CPU) {
        throw std::runtime_error(to_string(t.device().type) + " backend kernel is unavailable for this operation");
    }
}

std::vector<int64_t> strides_for(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

} // namespace llm
