#include "function.h"
#include "tensor.h"
#include <stdexcept>

// ============================================================
// AddBackward
// z = a + b  =>  dL/da = grad,  dL/db = grad
// ============================================================
void AddBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: 将 grad 分别 accumulate 到 inputs_[0] 和 inputs_[1]
    //   若对应 input 不 requires_grad 则跳过
    //   若 input 是非叶节点，还需递归调用 input->backward(local_grad)
}

// ============================================================
// SubBackward
// z = a - b  =>  dL/da = grad,  dL/db = -grad
// ============================================================
void SubBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: inputs_[0].accumulate_grad(*grad)
    //       neg_grad = -grad （逐元素取负）
    //       inputs_[1].accumulate_grad(*neg_grad)
    //       非叶节点递归 backward
}

// ============================================================
// MulBackward
// z = a * b  =>  dL/da = grad * b,  dL/db = grad * a
// ============================================================
void MulBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: grad_a = element-wise(grad * inputs_[1])
    //       grad_b = element-wise(grad * inputs_[0])
    //       分别 accumulate 并递归
}

// ============================================================
// DivBackward
// z = a / b  =>  dL/da = grad / b,  dL/db = -grad * a / b^2
// ============================================================
void DivBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: grad_a[i] = grad[i] / b[i]
    //       grad_b[i] = -grad[i] * a[i] / (b[i] * b[i])
    //       分别 accumulate 并递归
}

// ============================================================
// MatmulBackward
// z = A @ B  =>  dL/dA = grad @ B^T,  dL/dB = A^T @ grad
// ============================================================
void MatmulBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo:
    //   A = inputs_[0], B = inputs_[1]
    //   grad_A = grad.matmul(B.transpose())
    //   grad_B = A.transpose().matmul(grad)
    //   分别 accumulate 并递归
}

// ============================================================
// SumBackward
// z = sum(x)  =>  dL/dx[i] = grad（标量），broadcast 到 input_shape_
// ============================================================
void SumBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: 构造与 input_shape_ 相同的 Tensor，每个元素 = grad->data_[0]
    //       inputs_[0]->accumulate_grad(expanded_grad) 并递归
}

// ============================================================
// MeanBackward
// z = mean(x)  =>  dL/dx[i] = grad / n，broadcast 到 input_shape_
// ============================================================
void MeanBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: 与 SumBackward 类似，但每个元素 = grad->data_[0] / n_
}

// ============================================================
// ReluBackward
// z = relu(x)  =>  dL/dx[i] = grad[i] * (x[i] > 0 ? 1 : 0)
// ============================================================
void ReluBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: local_grad[i] = (inputs_[0]->data_[i] > 0) ? grad->data_[i] : 0
    //       inputs_[0]->accumulate_grad(local_grad) 并递归
}

// ============================================================
// SigmoidBackward
// z = sigmoid(x)  =>  dL/dx = grad * z * (1 - z)
// ============================================================
void SigmoidBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: local_grad[i] = grad[i] * output_[i] * (1 - output_[i])
    //       inputs_[0]->accumulate_grad(local_grad) 并递归
}

// ============================================================
// TanhBackward
// z = tanh(x)  =>  dL/dx = grad * (1 - z^2)
// ============================================================
void TanhBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: local_grad[i] = grad[i] * (1 - output_[i] * output_[i])
    //       inputs_[0]->accumulate_grad(local_grad) 并递归
}
