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
    // todo: 行优先（C-order）步长
    //   strides_[ndim-1] = 1
    //   strides_[i]      = strides_[i+1] * shape_[i+1]
    strides_.resize(shape_.size());
}

// ============================================================
// 构造
// ============================================================

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : shape_(std::move(shape)),
      requires_grad_(requires_grad),
      is_leaf_(true)
{
    data_.resize(numel(), 0.f);
    compute_strides();
}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad)
    : data_(std::move(data)),
      shape_(std::move(shape)),
      requires_grad_(requires_grad),
      is_leaf_(true)
{
    if ((int)data_.size() != numel())
        throw std::invalid_argument("data size does not match shape");
    compute_strides();
}

// ============================================================
// 工厂方法
// ============================================================

Tensor Tensor::zeros(std::vector<int> shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    // data_ 在构造里已全零
    return t;
}

Tensor Tensor::ones(std::vector<int> shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    std::fill(t.data_.begin(), t.data_.end(), 1.f);
    return t;
}

Tensor Tensor::randn(std::vector<int> shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    // todo: 用 std::mt19937 + std::normal_distribution<float>(0,1) 填充 t.data_
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
int Tensor::numel() const {
    if (shape_.empty()) return 0;
    int n = 1;
    for (int d : shape_) n *= d;
    return n;
}

//维度的数量
int Tensor::ndim() const {
    return (int)shape_.size();
}

std::string Tensor::shape_str() const {
    // todo: 拼接成 "(d0, d1, ...)" 格式的字符串
    return "";
}

// ============================================================
// 元素访问
// ============================================================

int Tensor::flat_index(const std::vector<int>& indices) const {
    // todo: 返回 sum(indices[i] * strides_[i])
    //   并做越界检查（indices[i] < 0 || indices[i] >= shape_[i] 时抛异常）
    return 0;
}

float& Tensor::at(const std::vector<int>& indices) {
    return data_[flat_index(indices)];
}

float Tensor::at(const std::vector<int>& indices) const {
    return data_[flat_index(indices)];
}

// ============================================================
// 形状变换
// ============================================================

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    // todo:
    //   1. 计算 new_shape 的 numel，检查与当前 numel() 相同
    //   2. 构造新 Tensor，共享同一份 data_（或拷贝，视实现简单程度决定）
    //   3. 重新计算 strides
    //   4. 不接入计算图（reshape 本身梯度为 identity，可暂不实现）
    return Tensor(shape_);  // placeholder
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    // todo:
    //   1. 拷贝 shape_ 和 strides_
    //   2. swap shape_[dim0] 与 shape_[dim1]
    //   3. swap strides_[dim0] 与 strides_[dim1]
    //   4. 共享 data_（不复制）
    return Tensor(shape_);  // placeholder
}

// ============================================================
// 数学运算
// ============================================================

// 通用辅助：element-wise 二元运算
// 返回新 Tensor 并挂上 grad_fn
static Tensor elementwise_op(
    const Tensor& a, const Tensor& b,
    std::function<float(float, float)> op,
    std::shared_ptr<Function> fn)
{
    if (a.shape_ != b.shape_)
        throw std::invalid_argument("shape mismatch in elementwise op");
    Tensor out(a.shape_);
    for (int i = 0; i < a.numel(); i++)
        out.data_[i] = op(a.data_[i], b.data_[i]);
    if (a.requires_grad_ || b.requires_grad_) {
        out.requires_grad_ = true;
        out.is_leaf_       = false;
        out.grad_fn_       = fn;
    }
    return out;
}

Tensor Tensor::operator+(const Tensor& other) const {
    auto fn = std::make_shared<AddBackward>();
    fn->inputs_ = {
        std::make_shared<Tensor>(*this),
        std::make_shared<Tensor>(other)
    };
    return elementwise_op(*this, other, [](float a, float b){ return a + b; }, fn);
}

Tensor Tensor::operator-(const Tensor& other) const {
    auto fn = std::make_shared<SubBackward>();
    fn->inputs_ = {
        std::make_shared<Tensor>(*this),
        std::make_shared<Tensor>(other)
    };
    return elementwise_op(*this, other, [](float a, float b){ return a - b; }, fn);
}

Tensor Tensor::operator*(const Tensor& other) const {
    auto fn = std::make_shared<MulBackward>();
    fn->inputs_ = {
        std::make_shared<Tensor>(*this),
        std::make_shared<Tensor>(other)
    };
    return elementwise_op(*this, other, [](float a, float b){ return a * b; }, fn);
}

Tensor Tensor::operator/(const Tensor& other) const {
    auto fn = std::make_shared<DivBackward>();
    fn->inputs_ = {
        std::make_shared<Tensor>(*this),
        std::make_shared<Tensor>(other)
    };
    return elementwise_op(*this, other, [](float a, float b){ return a / b; }, fn);
}

Tensor Tensor::matmul(const Tensor& other) const {
    // todo:
    //   1. 检查 ndim()==2 且 shape_[1] == other.shape_[0]
    //   2. 分配 result shape = {shape_[0], other.shape_[1]}，全零
    //   3. 三层循环：i, j, k: result[i][j] += this[i][k] * other[k][j]
    //   4. 构造 MatmulBackward，inputs_ = {this, other}，挂到 result.grad_fn_
    return Tensor({1});  // placeholder
}

Tensor Tensor::sum(int /*dim*/) const {
    // todo:
    //   dim == -1：对全部元素求和，返回 shape={1} 的标量 Tensor
    //   dim >= 0 ：沿该维度归约（可暂只实现全局版本）
    //   构造 SumBackward 并保存 input_shape_
    return Tensor({1});  // placeholder
}

Tensor Tensor::mean(int /*dim*/) const {
    // todo: 同 sum，但除以 n（或该维度大小）
    //   构造 MeanBackward 并保存 input_shape_ 和 n_
    return Tensor({1});  // placeholder
}

Tensor Tensor::max(int /*dim*/) const {
    // todo: 逐元素取最大（全局或沿 dim），暂不要求反向传播
    return Tensor({1});  // placeholder
}

Tensor Tensor::relu() const {
    Tensor out(shape_, requires_grad_);
    // todo: out.data_[i] = std::max(0.f, data_[i])
    if (requires_grad_) {
        out.is_leaf_  = false;
        auto fn = std::make_shared<ReluBackward>();
        fn->inputs_ = { std::make_shared<Tensor>(*this) };
        out.grad_fn_ = fn;
    }
    return out;
}

Tensor Tensor::sigmoid() const {
    Tensor out(shape_, requires_grad_);
    // todo: out.data_[i] = 1.f / (1.f + std::exp(-data_[i]))
    if (requires_grad_) {
        out.is_leaf_  = false;
        auto out_ptr = std::make_shared<Tensor>(out);
        auto fn = std::make_shared<SigmoidBackward>();
        fn->inputs_  = { std::make_shared<Tensor>(*this) };
        fn->output_  = out_ptr;
        out.grad_fn_ = fn;
    }
    return out;
}

Tensor Tensor::tanh_() const {
    Tensor out(shape_, requires_grad_);
    // todo: out.data_[i] = std::tanh(data_[i])
    if (requires_grad_) {
        out.is_leaf_  = false;
        auto out_ptr = std::make_shared<Tensor>(out);
        auto fn = std::make_shared<TanhBackward>();
        fn->inputs_  = { std::make_shared<Tensor>(*this) };
        fn->output_  = out_ptr;
        out.grad_fn_ = fn;
    }
    return out;
}

// ============================================================
// Autograd
// ============================================================

void Tensor::accumulate_grad(const Tensor& delta) {
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
}

void Tensor::zero_grad() {
    if (grad_)
        std::fill(grad_->data_.begin(), grad_->data_.end(), 0.f);
}

// ============================================================
// 工具
// ============================================================

void Tensor::print(const std::string& name) const {
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
