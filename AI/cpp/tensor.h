#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <string>

// 前向声明
struct Function;

// ============================================================
// Tensor：多维数组 + 自动微分支持
// ============================================================
class Tensor {
public:
    // ---- 数据 -----------------------------------------------
    std::vector<float>  data_;     // 扁平化连续存储
    std::vector<int>    shape_;    // 各维度大小，如 {2, 3}
    std::vector<int>    strides_;  // 各维度步长（元素个数）

    // ---- Autograd -------------------------------------------
    bool                            requires_grad_;
    std::shared_ptr<Tensor>         grad_;       // 梯度，与 this 同 shape
    std::shared_ptr<Function>       grad_fn_;    // 生成本节点的反向函数
    bool                            is_leaf_;    // 用户直接创建则为 true

    // =========================================================
    // 构造 / 析构
    // =========================================================

    // 用 shape 构造，data 全零；requires_grad 默认 false
    explicit Tensor(std::vector<int> shape, bool requires_grad = false);

    // 用已有数据 + shape 构造（data 会被拷贝）
    Tensor(std::vector<float> data, std::vector<int> shape,
           bool requires_grad = false);

    // 拷贝 / 移动
    /*
    * = default 告诉编译器自动生成拷贝构造函数，行为是对每个成员依次调用其自身的拷贝构造：

data_：std::vector<float> → vector 的拷贝构造会复制底层数组，深拷贝
shape_、strides_：同上，深拷贝
requires_grad_、is_leaf_：基本类型，直接复制值
grad_、grad_fn_：std::shared_ptr → 拷贝的是指针本身（引用计数 +1），不是深拷贝，两个 Tensor 共享同一个 grad 对象
     *
     */
    Tensor(const Tensor&)            = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&)                 = default;
    Tensor& operator=(Tensor&&)      = default;

    // =========================================================
    // 工厂方法
    // =========================================================
    static Tensor zeros(std::vector<int> shape, bool requires_grad = false);
    static Tensor ones (std::vector<int> shape, bool requires_grad = false);
    // todo: 用标准正态分布填充 data_，可使用 <random>
    static Tensor randn(std::vector<int> shape, bool requires_grad = false);
    // todo: 用均匀分布 [low, high) 填充 data_
    static Tensor rand (std::vector<int> shape, float low = 0.f, float high = 1.f,
                        bool requires_grad = false);

    // =========================================================
    // 基础属性
    // =========================================================
    int  numel() const;          // 元素总数 = 所有维度之积
    int  ndim()  const;          // 维度数 = shape_.size()
    // todo: 返回人类可读的 shape 字符串，如 "(2, 3)"
    std::string shape_str() const;

    // =========================================================
    // 元素访问
    // =========================================================
    // 用多维下标访问（读写），indices 长度须等于 ndim()
    float& at(const std::vector<int>& indices);
    float  at(const std::vector<int>& indices) const;

    // 将多维下标转为 data_ 的一维下标（利用 strides_）
    // todo: 实现：offset = sum(indices[i] * strides_[i])
    int flat_index(const std::vector<int>& indices) const;

    // =========================================================
    // 形状变换（返回新 Tensor，与原 Tensor 共享 data_ 时需注意）
    // =========================================================
    // todo: 检查 numel 不变后重新计算 strides，返回新 Tensor（共享 data_）
    Tensor reshape(std::vector<int> new_shape) const;

    // todo: 交换 dim0 与 dim1 对应的 shape 和 strides，不复制数据
    Tensor transpose(int dim0 = 0, int dim1 = 1) const;

    // =========================================================
    // 数学运算（均会将当前节点接入计算图）
    // =========================================================
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // element-wise
    Tensor operator/(const Tensor& other) const;

    // todo: 矩阵乘法（仅支持 2D），result[i][j] = sum_k(A[i][k] * B[k][j])
    Tensor matmul(const Tensor& other) const;

    // 归约运算
    // todo: 对所有元素求和（dim=-1 表示全局），返回标量 Tensor 或沿指定维度归约
    Tensor sum(int dim = -1) const;
    // todo: 同上，求均值
    Tensor mean(int dim = -1) const;
    // todo: 逐元素取最大值，返回同 shape Tensor
    Tensor max(int dim = -1) const;

    // 激活函数
    // todo: ReLU：max(0, x)，逐元素
    Tensor relu() const;
    // todo: Sigmoid：1 / (1 + exp(-x))，逐元素
    Tensor sigmoid() const;
    // todo: Tanh：(exp(x)-exp(-x))/(exp(x)+exp(-x))，逐元素
    Tensor tanh_() const;

    // =========================================================
    // Autograd
    // =========================================================
    // 从标量 Tensor 出发，反向传播梯度
    // todo: 调用 grad_fn_->backward(upstream_grad) 并沿计算图递归传播
    void backward(std::shared_ptr<Tensor> upstream_grad = nullptr);

    // 将 grad_ 清零（不销毁）；叶节点常用
    void zero_grad();

    // 内部辅助：将 other 的梯度累加到 grad_（不存在则先创建全零 grad_）
    void accumulate_grad(const Tensor& delta);

    // =========================================================
    // 工具
    // =========================================================
    void print(const std::string& name = "") const;

private:
    // 根据 shape_ 重新计算 strides_（行优先 / C-order）
    // strides_[i] = product(shape_[i+1 .. end])
    // todo: 实现行优先步长计算
    void compute_strides();
};
