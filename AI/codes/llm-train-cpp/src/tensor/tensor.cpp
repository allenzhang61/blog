#include "llm/tensor.hpp"

namespace llm {

TensorCudaStorage::~TensorCudaStorage() {
    if (release != nullptr) {
        release(*this);
    }
}

namespace {

void sync_data_to_host(TensorNode& node) {
    TensorCudaStorage* storage = nullptr;
    const char* name = nullptr;
    if (node.device.type == DeviceType::CUDA) {
        storage = node.cuda_storage.get();
        name = "CUDA";
    } else if (node.device.type == DeviceType::Metal) {
        storage = node.metal_storage.get();
        name = "Metal";
    }
    if (storage == nullptr || !node.device_data_dirty) {
        return;
    }
    if (storage->copy_data_to_host == nullptr) {
        throw std::runtime_error(std::string(name) + " tensor data sync to host is unavailable");
    }
    storage->copy_data_to_host(*storage, node.data);
    node.device_data_dirty = false;
}

void sync_grad_to_host(TensorNode& node) {
    TensorCudaStorage* storage = nullptr;
    const char* name = nullptr;
    if (node.device.type == DeviceType::CUDA) {
        storage = node.cuda_storage.get();
        name = "CUDA";
    } else if (node.device.type == DeviceType::Metal) {
        storage = node.metal_storage.get();
        name = "Metal";
    }
    if (storage == nullptr || !node.device_grad_dirty) {
        return;
    }
    if (storage->copy_grad_to_host == nullptr) {
        throw std::runtime_error(std::string(name) + " tensor grad sync to host is unavailable");
    }
    storage->copy_grad_to_host(*storage, node.grad);
    node.device_grad_dirty = false;
}

} // namespace

Tensor::Tensor() : node(std::make_shared<TensorNode>()) {
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device, bool requires_grad)
    : node(std::make_shared<TensorNode>()) {
    node->shape = std::move(shape);
    node->dtype = dtype;
    node->device = device;
    node->requires_grad = requires_grad;
    node->data.assign(product(node->shape), 0.0);
    if (requires_grad) {
        node->grad.assign(node->data.size(), 0.0);
    }
    if (node->device.type == DeviceType::CUDA || node->device.type == DeviceType::Metal) {
        node->host_data_dirty = true;
        node->host_grad_dirty = requires_grad;
    }
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
    if (requires_grad) {
        node->grad.assign(node->data.size(), 0.0);
    }
    if (node->device.type == DeviceType::CUDA || node->device.type == DeviceType::Metal) {
        node->host_data_dirty = true;
        node->host_grad_dirty = requires_grad;
    }
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
    for (auto& v : t.data()) {
        v = dist(gen);
    }
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

const std::vector<int64_t>& Tensor::shape() const {
    return node->shape;
}

DType Tensor::dtype() const {
    return node->dtype;
}

Device Tensor::device() const {
    return node->device;
}

bool Tensor::requires_grad() const {
    return node->requires_grad;
}

int64_t Tensor::numel() const {
    return static_cast<int64_t>(node->data.size());
}

std::vector<double>& Tensor::data() {
    return mutable_data();
}

const std::vector<double>& Tensor::data() const {
    sync_data_to_host(*node);
    return node->data;
}

std::vector<double>& Tensor::mutable_data() {
    sync_data_to_host(*node);
    mark_data_host_dirty();
    return node->data;
}

std::vector<double>& Tensor::grad() const {
    return mutable_grad();
}

std::vector<double>& Tensor::mutable_grad() const {
    sync_grad_to_host(*node);
    if (node->grad.empty()) {
        node->grad.assign(node->data.size(), 0.0);
    }
    mark_grad_host_dirty();
    return node->grad;
}

void Tensor::mark_data_host_dirty() const {
    if (node->device.type == DeviceType::CUDA || node->device.type == DeviceType::Metal) {
        node->host_data_dirty = true;
        node->device_data_dirty = false;
    }
}

void Tensor::mark_grad_host_dirty() const {
    if (node->device.type == DeviceType::CUDA || node->device.type == DeviceType::Metal) {
        node->host_grad_dirty = true;
        node->device_grad_dirty = false;
    }
}

double Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() requires scalar tensor");
    }
    sync_data_to_host(*node);
    return node->data[0];
}

void Tensor::zero_grad() {
    node->grad.assign(node->data.size(), 0.0);
    if (node->device.type == DeviceType::CUDA || node->device.type == DeviceType::Metal) {
        TensorCudaStorage* storage = node->device.type == DeviceType::CUDA ? node->cuda_storage.get()
                                                                           : node->metal_storage.get();
        if (storage && storage->fill_grad) {
            storage->fill_grad(*storage, node->grad.size(), 0.0f);
            node->host_grad_dirty = false;
            node->device_grad_dirty = true;
        } else {
            node->host_grad_dirty = true;
            node->device_grad_dirty = false;
        }
    }
}

namespace {
void topo_visit(const Tensor& t, std::unordered_map<TensorNode*, bool>& seen, std::vector<Tensor>& order) {
    TensorNode* ptr = t.node.get();
    if (seen[ptr]) {
        return;
    }
    seen[ptr] = true;
    for (const auto& parent : t.node->parents) {
        topo_visit(parent, seen, order);
    }
    order.push_back(t);
}
} // namespace

void Tensor::backward() {
    if (numel() != 1) {
        throw std::runtime_error("backward() expects a scalar tensor");
    }
    std::unordered_map<TensorNode*, bool> seen;
    std::vector<Tensor> order;
    topo_visit(*this, seen, order);
    grad().assign(1, 1.0);
    TensorCudaStorage* storage = nullptr;
    if (node->device.type == DeviceType::CUDA) {
        storage = node->cuda_storage.get();
    } else if (node->device.type == DeviceType::Metal) {
        storage = node->metal_storage.get();
    }
    if (storage && storage->copy_grad_from_host) {
        storage->copy_grad_from_host(*storage, node->grad);
        node->host_grad_dirty = false;
        node->device_grad_dirty = true;
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (it->node->backward_fn) {
            it->node->backward_fn();
        }
    }
}

std::vector<int64_t> strides_for(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

} // namespace llm
