#pragma once
#include <vector>
#include <memory>

class Tensor;

// ============================================================
// Function：计算图中的一个运算节点
//   forward()  已在 Tensor 的运算符里完成
//   backward() 接收上游梯度，计算并向输入节点传播
// ============================================================
struct Function {
    // 梯度传播目标。叶子节点引用原始 Tensor；非叶节点保存一份带 grad_fn_ 的节点副本。
    std::vector<std::shared_ptr<Tensor>> next_;

    // forward 时按需保存的值拷贝，用于计算局部梯度。
    // 例如 MulBackward 需要 forward 当时的 a、b；AddBackward 不需要保存。
    std::vector<std::shared_ptr<Tensor>> saved_tensors_;

    virtual ~Function() = default;

    virtual void backward(std::shared_ptr<Tensor> grad) = 0;
};

// ============================================================
// 各运算的反向节点
// ============================================================

// z = a + b  => dL/da = grad, dL/db = grad
struct AddBackward : public Function {
    // todo: 实现 backward：grad 直接传给两个输入
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = a - b  => dL/da = grad, dL/db = -grad
struct SubBackward : public Function {
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = a * b  (element-wise)  => dL/da = grad*b, dL/db = grad*a
struct MulBackward : public Function {
    // 需要保存 forward 时的 a, b 用于求偏导
    // saved_tensors_[0] = a, saved_tensors_[1] = b
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = a / b  => dL/da = grad/b, dL/db = -grad*a/b^2
struct DivBackward : public Function {
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = A @ B  (matmul 2D)  => dL/dA = grad @ B^T, dL/dB = A^T @ grad
struct MatmulBackward : public Function {
    // todo: 实现 backward（注意矩阵转置）
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = sum(x)  => dL/dx = grad broadcast 到 x 的 shape（全 1 矩阵 * grad）
struct SumBackward : public Function {
    std::vector<int> input_shape_;   // 保存输入 shape 用于 broadcast
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = mean(x)  => dL/dx = grad / n broadcast 到 x 的 shape
struct MeanBackward : public Function {
    std::vector<int> input_shape_;
    int n_;
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = relu(x)  => dL/dx = grad * (x > 0 ? 1 : 0)
struct ReluBackward : public Function {
    // saved_tensors_[0] = x（forward 时的输入，用于判断正负）
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = sigmoid(x)  => dL/dx = grad * z * (1 - z)
struct SigmoidBackward : public Function {
    std::shared_ptr<Tensor> output_;  // 保存 forward 的输出 z
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};

// z = tanh(x)  => dL/dx = grad * (1 - z^2)
struct TanhBackward : public Function {
    std::shared_ptr<Tensor> output_;
    // todo: 实现 backward
    void backward(std::shared_ptr<Tensor> grad) override;
};
