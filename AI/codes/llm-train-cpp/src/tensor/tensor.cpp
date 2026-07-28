#include "llm/tensor.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace llm {

TensorStorage::~TensorStorage() {
    if (release != nullptr) {
        release(*this);
    }
}

namespace {

void sync_data_to_host(TensorNode& node) {
    TensorStorage* storage = nullptr;
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
    TensorStorage* storage = nullptr;
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
    // static_cast<int64_t>(...) ：把 size_t 转成 int64_t ，让两边 类型一致再比较 ，
    // 避免有符号/无符号混比引发的编译告警和错误比较。
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
    //todo 这个应该放在函数初始化阶段
    //
    // 固定种子 123 ：结果是 确定性/可复现 的——同样的程序每次运行得到相同的随机初始化。
    // 对教学和调试友好（测试可断言具体数值），但生产训练通常需要可配置种子或真随机源。
    //
    // static 引擎非线程安全 ：多线程同时调用 randn 会有数据竞争。
    // 当前项目是单线程 CPU/单命令流，无碍；若将来并行则需加锁或改为线程局部引擎。
    static std::mt19937 gen(123);
    // std::normal_distribution<double> dist(0.0, scale) ：正态分布 分布器 ，
    // 把引擎的原始随机数映射成均值 0.0 、标准差 scale 的高斯分布。
    std::normal_distribution<double> dist(0.0, scale);
    for (auto& v : t.data()) {
        v = dist(gen);
    }
    return t;
}

Tensor Tensor::uniform(const std::vector<int64_t>& shape, double bound, Device device, bool requires_grad) {
    Tensor t(shape, DType::Float32, device, requires_grad);
    // 函数内 static 局部引擎：首次调用 uniform 时以种子 456 初始化一次，
    // 之后所有调用共用这一个实例并持续推进随机序列，保证初始化确定性/可复现。
    // 与 randn 里的 gen(123) 是相互独立的两个引擎（各用各的种子）。
    static std::mt19937 gen(456);
    std::uniform_real_distribution<double> dist(-bound, bound);
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
    if (node->shape.empty() && node->data.empty() && !node->cuda_storage && !node->metal_storage) {
        return 0;
    }
    return product(node->shape);
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
    mark_data_host_dirty(); // mark_data_host_dirty() 表示"host 端数据被改了，设备端需要重新同步"
    return node->data;
}

std::vector<double>& Tensor::grad() const {
    return mutable_grad();
}

std::vector<double>& Tensor::mutable_grad() const {
    sync_grad_to_host(*node);
    if (node->grad.empty()) {
        node->grad.assign(static_cast<size_t>(numel()), 0.0);
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
    // 先把 host 端梯度缓冲区全部清零（大小与数据一致）。
    node->grad.assign(static_cast<size_t>(numel()), 0.0);
    // 对于 GPU 设备，还需要同步清零设备端的梯度缓冲区。
    if (node->device.type == DeviceType::CUDA || node->device.type == DeviceType::Metal) {
        // 根据设备类型取对应的设备存储。
        TensorStorage* storage = node->device.type == DeviceType::CUDA ? node->cuda_storage.get()
                                                                           : node->metal_storage.get();
        if (storage && storage->fill_grad) {
            // 设备端已分配且支持 fill_grad：直接在设备上把梯度填 0，
            // 此时设备端是最新的（device 为脏、host 干净），避免多余的 host->device 拷贝。
            storage->fill_grad(*storage, static_cast<size_t>(numel()), 0.0f);
            node->host_grad_dirty = false;
            node->device_grad_dirty = true;
        } else {
            // 设备端尚未就绪：仅 host 端清了零，标记 host 为脏，
            // 待下次同步时再把 host 的零梯度推到设备端。
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
    // 反向传播必须从标量（loss）出发，否则梯度无从定义。
    if (numel() != 1) {
        throw std::runtime_error("backward() expects a scalar tensor");
    }
    // 对计算图做拓扑排序，order 中前驱在前、后继在后。
    std::unordered_map<TensorNode*, bool> seen;
    std::vector<Tensor> order;
    topo_visit(*this, seen, order);
    // 种子梯度：标量对自身的导数为 1。
    grad().assign(1, 1.0);
    // 若在 GPU 上，把这个种子梯度同步到设备端。
    TensorStorage* storage = nullptr;
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
    // 按拓扑逆序依次执行各节点的反向函数，把梯度沿计算图回传。
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (it->node->backward_fn) {
            it->node->backward_fn();
        }
    }
    // 反向传播完成后主动打断计算图，避免内存泄漏。
    //
    // 每个中间节点的 backward_fn 闭包按值捕获了输出张量自身（含 shared_ptr<TensorNode>），
    // 形成 node -> backward_fn -> 捕获的 Tensor -> node 的自引用循环，
    // 导致整张图的引用计数永不归零、无法析构。训练时每步都新建一整张大图，
    // 若不清理会在几十步内累积到数十 GB 内存。
    //
    // 本项目不做二次反向，backward 结束后图即可丢弃：清空各节点的 backward_fn 和 parents，
    // 循环引用被打断，中间节点随 order 离开作用域而正常释放；叶子（参数）节点的
    // shared_ptr 仍由模型持有，其 data/grad 不受影响。
    for (auto& t : order) {
        t.node->backward_fn = nullptr;
        t.node->parents.clear();
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
