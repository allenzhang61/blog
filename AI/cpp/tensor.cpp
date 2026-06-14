#include "tensor.h"
#include "function.h"
#include <stdexcept>
#include <numeric>
#include <random>
#include <iostream>
#include <sstream>
#include <cmath>

// ============================================================
// 私有辅助
// ============================================================

void Tensor::compute_strides() {
    //done
    // 行优先（C-order）步长
    //   strides_[ndim-1] = 1
    //   strides_[i]      = strides_[i+1] * shape_[i+1]
    strides_.resize(shape_.size());
    strides_[ndim() - 1] = 1;
    for (int i = ndim() - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// ============================================================
// 构造
// ============================================================

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : shape_(std::move(shape)),
      requires_grad_(requires_grad),
      is_leaf_(true) {
    // resize(n, val) 把 vector 的大小调整为 n，新增的元素用 val 填充
    // 因为是在构造函数中，vector 是一个全新的空 vector
    // 实际就是用 val 对 vector 进行了初始化
    data_.resize(numel(), 0.f);
    compute_strides();
}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad)
    : data_(std::move(data)),
      shape_(std::move(shape)),
      requires_grad_(requires_grad),
      is_leaf_(true) {
    if ((int) data_.size() != numel())
        throw std::invalid_argument("data size does not match shape");
    compute_strides();
}

// ============================================================
// 工厂方法
// ============================================================

//done
Tensor Tensor::zeros(std::vector<int> shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    // data_ 在构造里已全零
    return t;
}

//done
Tensor Tensor::ones(std::vector<int> shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    std::fill(t.data_.begin(), t.data_.end(), 1.f);
    return t;
}

//done
Tensor Tensor::randn(std::vector<int> shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    // 用 std::mt19937 + std::normal_distribution<float>(0,1) 填充 t.data_
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.f, 1.f);
    for (float &x: t.data_) x = dist(rng);
    return t;
}

Tensor Tensor::rand(std::vector<int> shape, float low, float high, bool requires_grad) {
    Tensor t(shape, requires_grad);
    // todo: 用 std::mt19937 + std::uniform_real_distribution<float>(low,high) 填充 t.data_
    return t;
}

// ============================================================
// 基础属性
// ============================================================

// 元素的数量
//done
int Tensor::numel() const {
    // if (shape_.empty()) return 0;
    int n = 1;
    for (int d: shape_) n *= d;
    return n;
}

//维度的数量
// done
int Tensor::ndim() const {
    return (int) shape_.size();
}

std::string Tensor::shape_str() const {
    // todo: 拼接成 "(d0, d1, ...)" 格式的字符串
    return "";
}

// ============================================================
// 元素访问
// ============================================================
//done
int Tensor::flat_index(const std::vector<int> &indices) const {
    // 返回 sum(indices[i] * strides_[i])
    //   并做越界检查（indices[i] < 0 || indices[i] >= shape_[i] 时抛异常）
    if (indices.size() != strides_.size())
        throw std::invalid_argument("indices size does not match shape");

    int s = 0;
    for (int i = 0; i < strides_.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("indices out of bounds");
        }

        s += strides_[i] * indices[i];
    }
    return s;
}

//done
float &Tensor::at(const std::vector<int> &indices) {
    return data_[flat_index(indices)];
}

float Tensor::at(const std::vector<int> &indices) const {
    return data_[flat_index(indices)];
}

// ============================================================
// 形状变换
// ============================================================

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    //   1. 计算 new_shape 的 numel，检查与当前 numel() 相同
    //   2. 构造新 Tensor，共享同一份 data_（或拷贝，视实现简单程度决定）
    //   3. 重新计算 strides
    //   4. 不接入计算图（reshape 本身梯度为 identity，可暂不实现）
    int new_numel = 1;
    for (int d: new_shape) new_numel *= d;
    if (new_numel != numel()) {
        throw std::invalid_argument("number of elements does not match shape");
    }

    return Tensor(data_, new_shape, requires_grad_);

    // return Tensor(shape_); // placeholder
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    //   1. 拷贝 shape_ 和 strides_
    //   2. swap shape_[dim0] 与 shape_[dim1]
    //   3. swap strides_[dim0] 与 strides_[dim1]
    //   4. 共享 data_（不复制）
    Tensor result(data_, shape_, requires_grad_);
    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.strides_[dim0], result.strides_[dim1]);
    return result;

    // return Tensor(shape_); // placeholder
}

// ============================================================
// 数学运算
// ============================================================

// 通用辅助：element-wise 二元运算
// 返回新 Tensor 并挂上 grad_fn
static Tensor elementwise_op(
    const Tensor &a, const Tensor &b,
    std::function<float(float, float)> op,
    std::shared_ptr<Function> fn) {
    if (a.shape_ != b.shape_)
        throw std::invalid_argument("shape mismatch in elementwise op");
    Tensor out(a.shape_);
    for (int i = 0; i < a.numel(); i++)
        out.data_[i] = op(a.data_[i], b.data_[i]);
    if (a.requires_grad_ || b.requires_grad_) {
        out.requires_grad_ = true;
        out.is_leaf_ = false;
        out.grad_fn_ = fn;
    }
    return out;
}

static std::shared_ptr<Tensor> autograd_edge(const Tensor& t) {
    if (t.is_leaf_)
        return std::shared_ptr<Tensor>(const_cast<Tensor*>(&t), [](Tensor*) {});
    return std::make_shared<Tensor>(t);
}

static std::shared_ptr<Tensor> save_for_backward(const Tensor& t) {
    return std::make_shared<Tensor>(t);
}

Tensor Tensor::operator+(const Tensor &other) const {
    auto fn = std::make_shared<AddBackward>();
    fn->next_ = { autograd_edge(*this), autograd_edge(other) };
    return elementwise_op(*this, other, [](float a, float b) { return a + b; }, fn);
}

Tensor Tensor::operator-(const Tensor &other) const {
    auto fn = std::make_shared<SubBackward>();
    fn->next_ = { autograd_edge(*this), autograd_edge(other) };
    return elementwise_op(*this, other, [](float a, float b) { return a - b; }, fn);
}

Tensor Tensor::operator*(const Tensor &other) const {
    auto fn = std::make_shared<MulBackward>();
    fn->next_ = { autograd_edge(*this), autograd_edge(other) };
    fn->saved_tensors_ = { save_for_backward(*this), save_for_backward(other) };
    return elementwise_op(*this, other, [](float a, float b) { return a * b; }, fn);
}

Tensor Tensor::operator/(const Tensor &other) const {
    auto fn = std::make_shared<DivBackward>();
    fn->next_ = { autograd_edge(*this), autograd_edge(other) };
    fn->saved_tensors_ = { save_for_backward(*this), save_for_backward(other) };
    return elementwise_op(*this, other, [](float a, float b) { return a / b; }, fn);
}

Tensor Tensor::matmul(const Tensor &other) const {
    //   1. 检查 ndim()==2 且 shape_[1] == other.shape_[0]
    //   2. 分配 result shape = {shape_[0], other.shape_[1]}，全零
    //   3. 三层循环：i, j, k: result[i][j] += this[i][k] * other[k][j]
    //   4. 构造 MatmulBackward，将输入接入计算图并按需保存 forward 值
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("number of dimensions do not match");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("number of shapes do not match");
    }
    Tensor out({shape_[0], other.shape_[1]});
    // 直接用 strides_ 做下标运算，避免 at() 每次 vector 堆分配
    for (int i = 0; i < shape_[0]; i++) {
        for (int j = 0; j < other.shape_[1]; j++) {
            float s = 0;
            for (int k = 0; k < shape_[1]; k++) {
                s += data_[i * strides_[0] + k * strides_[1]]
                   * other.data_[k * other.strides_[0] + j * other.strides_[1]];
            }
            out.data_[i * out.strides_[0] + j * out.strides_[1]] = s;
        }
    }
    if (requires_grad_ || other.requires_grad_) {
        out.requires_grad_ = true;
        out.is_leaf_ = false;
        auto fn = std::make_shared<MatmulBackward>();
        fn->next_ = { autograd_edge(*this), autograd_edge(other) };
        fn->saved_tensors_ = { save_for_backward(*this), save_for_backward(other) };
        out.grad_fn_ = fn;
    }
    return out;
    // return Tensor({1}); // placeholder
}

Tensor Tensor::sum(int dim) const {
    // todo:
    //   dim == -1：对全部元素求和，返回 shape={1} 的标量 Tensor
    //   dim >= 0 ：沿该维度归约（可暂只实现全局版本）
    //   构造 SumBackward 并保存 input_shape_
    float s = 0;
    if (dim == -1) {
        for (float d: data_) { s += d; }
    } else {
        //todo dim >=0
        throw std::invalid_argument("dim should be -1");
    }
    Tensor out({1});
    out.data_[0] = s;
    if (requires_grad_) {
        out.requires_grad_ = true;
        out.is_leaf_ = false;
        auto fn = std::make_shared<SumBackward>();
        fn->next_ = { autograd_edge(*this) };
        fn->input_shape_ = shape_;
        out.grad_fn_ = fn;
    }
    return out;
    // return Tensor({1}); // placeholder
}

Tensor Tensor::mean(int dim) const {
    // todo: 同 sum，但除以 n（或该维度大小）
    //   构造 MeanBackward 并保存 input_shape_ 和 n_
    float s = 0;
    if (dim == -1) {
        for (float d: data_) { s += d; }
        s /= numel();
    } else {
        //todo dim>=0
        throw std::invalid_argument("dim should be -1");
    }

    Tensor out({1});
    out.data_[0] = s;

    if (requires_grad_) {
        out.requires_grad_ = true;
        out.is_leaf_ = false;
        auto fn = std::make_shared<MeanBackward>();
        fn->next_        = { autograd_edge(*this) };
        fn->input_shape_ = shape_;
        fn->n_           = numel();
        out.grad_fn_     = fn;
    }

    return out;

    // return Tensor({1}); // placeholder
}

Tensor Tensor::max(int dim) const {
    // todo: 逐元素取最大（全局或沿 dim），暂不要求反向传播
    if (numel() == 0) {
        throw std::invalid_argument("number of elements should be 0");
    }
    float s = data_[0];
    if (dim == -1) {
        for (float d: data_) { s = std::max(s, d); }
    } else {
        //todo dim>=0
        throw std::invalid_argument("dim should be -1");
    }
    Tensor out({1});
    out.data_[0] = s;
    return out;
    // return Tensor({1}); // placeholder
}

Tensor Tensor::relu() const {
    Tensor out(shape_, requires_grad_);
    // todo: out.data_[i] = std::max(0.f, data_[i])
    for (int i = 0; i < numel(); i++) {
        out.data_[i] = std::max(data_[i], 0.f);
    }
    if (requires_grad_) {
        out.is_leaf_ = false;
        auto fn = std::make_shared<ReluBackward>();
        fn->next_ = { autograd_edge(*this) };
        fn->saved_tensors_ = { save_for_backward(*this) };
        out.grad_fn_ = fn;
    }
    return out;
}

Tensor Tensor::sigmoid() const {
    Tensor out(shape_, requires_grad_);
    for (int i = 0; i < numel(); i++)
        out.data_[i] = 1.f / (1.f + std::exp(-data_[i]));
    if (requires_grad_) {
        out.is_leaf_ = false;
        auto fn = std::make_shared<SigmoidBackward>();
        fn->next_   = { autograd_edge(*this) };
        fn->output_ = std::make_shared<Tensor>(out);
        out.grad_fn_ = fn;
    }
    return out;
}

Tensor Tensor::tanh_() const {
    Tensor out(shape_, requires_grad_);
    for (int i = 0; i < numel(); i++)
        out.data_[i] = std::tanh(data_[i]);
    if (requires_grad_) {
        out.is_leaf_ = false;
        auto fn = std::make_shared<TanhBackward>();
        fn->next_   = { autograd_edge(*this) };
        fn->output_ = std::make_shared<Tensor>(out);
        out.grad_fn_ = fn;
    }
    return out;
}

// ============================================================
// Autograd
// ============================================================

void Tensor::accumulate_grad(const Tensor &delta) {
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_);
    }
    for (int i = 0; i < numel(); i++)
        grad_->data_[i] += delta.data_[i];
}

void Tensor::backward(std::shared_ptr<Tensor> upstream_grad) {
    // todo:
    //   1. 若 upstream_grad 为空（从标量调用），初始化为全 1、shape={1}
    //   2. 若 is_leaf_ 且 requires_grad_，调用 accumulate_grad(*upstream_grad) 后 return
    //   3. 否则调用 grad_fn_->backward(upstream_grad)（grad_fn_ 内部会递归调 inputs 的 backward）
    if (upstream_grad == nullptr) {
        upstream_grad = std::make_shared<Tensor>(std::vector<float>{1.f}, std::vector<int>{1});
    }
    //叶子节点
    if (is_leaf_) {
        if (requires_grad_) {
            accumulate_grad(*upstream_grad);
        }
        return;
    }
    //非叶子节点
    grad_fn_->backward(upstream_grad);
}

void Tensor::zero_grad() {
    if (grad_)
        std::fill(grad_->data_.begin(), grad_->data_.end(), 0.f);
}

// ============================================================
// 工具
// ============================================================

void Tensor::print(const std::string &name) const {
    if (!name.empty()) std::cout << name << " ";
    // todo: 递归或迭代地按 shape_ 打印多维数组，类似 numpy 的输出格式
    //   简单实现：直接打印扁平 data_，标注 shape
    std::cout << "Tensor" << shape_str() << " [";
    for (int i = 0; i < numel(); i++) {
        std::cout << data_[i];
        if (i + 1 < numel()) std::cout << ", ";
    }
    std::cout << "]\n";
}
