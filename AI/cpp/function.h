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
    // 保存 forward 时的输入（weak_ptr 避免循环引用）
    // 但为简单起见，这里直接存 shared_ptr，实际可按需改为 weak_ptr
    std::vector<std::shared_ptr<Tensor>> inputs_;

    virtual ~Function() = default;

    // 接收来自下游的梯度 grad（与 output 同 shape），
    // 计算各 input 的梯度并调用 input->accumulate_grad()
    // todo: 每个子类必须实现此方法，写出对应运算的偏导数公式
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
    // inputs_[0] = a, inputs_[1] = b
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
    // inputs_[0] = x（forward 时的输入，用于判断正负）
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
