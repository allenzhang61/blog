#include "function.h"
#include "tensor.h"
#include <stdexcept>

// ============================================================
// AddBackward
// z = a + b  =>  dL/da = grad,  dL/db = grad
// ============================================================
void AddBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: 将 grad 分别 accumulate 到 next_[0] 和 next_[1]
    //   若对应 input 不 requires_grad 则跳过
    //   若 input 是非叶节点，还需递归调用 input->backward(local_grad)
    // z = a + b，dL/da = grad，dL/db = grad，直接把上游梯度传给两个输入
    for (auto& input : next_) {
        if (input->requires_grad_)
            input->backward(grad);
    }
}

// ============================================================
// SubBackward
// z = a - b  =>  dL/da = grad,  dL/db = -grad
// ============================================================
void SubBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: next_[0].accumulate_grad(*grad)
    //       neg_grad = -grad （逐元素取负）
    //       next_[1].accumulate_grad(*neg_grad)
    //       非叶节点递归 backward
    if (next_[0]->requires_grad_)
        next_[0]->backward(grad);

    if (next_[1]->requires_grad_) {
        Tensor neg_grad(grad->shape_);
        for (int i = 0; i < grad->numel(); i++)
            neg_grad.data_[i] = -grad->data_[i];
        next_[1]->backward(std::make_shared<Tensor>(neg_grad));
    }
}

// ============================================================
// MulBackward
// z = a * b  =>  dL/da = grad * b,  dL/db = grad * a
// ============================================================
void MulBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: grad_a = element-wise(grad * saved_tensors_[1])
    //       grad_b = element-wise(grad * saved_tensors_[0])
    //       分别 accumulate 并递归
    // saved_tensors_[0] = copy of a，saved_tensors_[1] = copy of b（用于读取 forward 值）
    if (next_[0]->requires_grad_) {
        Tensor grad_a(grad->shape_);
        for (int i = 0; i < grad->numel(); i++)
            grad_a.data_[i] = grad->data_[i] * saved_tensors_[1]->data_[i];
        next_[0]->backward(std::make_shared<Tensor>(grad_a));
    }
    if (next_[1]->requires_grad_) {
        Tensor grad_b(grad->shape_);
        for (int i = 0; i < grad->numel(); i++)
            grad_b.data_[i] = grad->data_[i] * saved_tensors_[0]->data_[i];
        next_[1]->backward(std::make_shared<Tensor>(grad_b));
    }
}

// ============================================================
// DivBackward
// z = a / b  =>  dL/da = grad / b,  dL/db = -grad * a / b^2
// ============================================================
void DivBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: grad_a[i] = grad[i] / b[i]
    //       grad_b[i] = -grad[i] * a[i] / (b[i] * b[i])
    //       分别 accumulate 并递归
    if (next_[0]->requires_grad_) {
        Tensor grad_a(grad->shape_);
        for (int i = 0; i < grad->numel(); i++)
            grad_a.data_[i] = grad->data_[i] / saved_tensors_[1]->data_[i];
        next_[0]->backward(std::make_shared<Tensor>(grad_a));
    }
    if (next_[1]->requires_grad_) {
        Tensor grad_b(grad->shape_);
        for (int i = 0; i < grad->numel(); i++) {
            float b = saved_tensors_[1]->data_[i];
            grad_b.data_[i] = -grad->data_[i] * saved_tensors_[0]->data_[i] / (b * b);
        }
        next_[1]->backward(std::make_shared<Tensor>(grad_b));
    }
}

// ============================================================
// MatmulBackward
// z = A @ B  =>  dL/dA = grad @ B^T,  dL/dB = A^T @ grad
// ============================================================
void MatmulBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo:
    //   A = saved_tensors_[0], B = saved_tensors_[1]
    //   grad_A = grad.matmul(B.transpose())
    //   grad_B = A.transpose().matmul(grad)
    //   分别 accumulate 并递归
    auto& A = saved_tensors_[0];  // copy of A，shape (M, K)
    auto& B = saved_tensors_[1];  // copy of B，shape (K, N)
    // grad shape: (M, N)

    if (next_[0]->requires_grad_) {
        // dL/dA = grad @ B^T，shape (M, K)
        Tensor grad_A = grad->matmul(B->transpose());
        next_[0]->backward(std::make_shared<Tensor>(grad_A));
    }
    if (next_[1]->requires_grad_) {
        // dL/dB = A^T @ grad，shape (K, N)
        Tensor grad_B = A->transpose().matmul(*grad);
        next_[1]->backward(std::make_shared<Tensor>(grad_B));
    }
}

// ============================================================
// SumBackward
// z = sum(x)  =>  dL/dx[i] = grad（标量），broadcast 到 input_shape_
// ============================================================
void SumBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: 构造与 input_shape_ 相同的 Tensor，每个元素 = grad->data_[0]
    //       next_[0]->accumulate_grad(expanded_grad) 并递归
    Tensor expanded(input_shape_);
    for (float& x : expanded.data_)
        x = grad->data_[0];
    auto ep = std::make_shared<Tensor>(expanded);
    if (next_[0]->requires_grad_)
        next_[0]->backward(ep);
}

// ============================================================
// MeanBackward
// z = mean(x)  =>  dL/dx[i] = grad / n，broadcast 到 input_shape_
// ============================================================
void MeanBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: 与 SumBackward 类似，但每个元素 = grad->data_[0] / n_
    //       next_[0]->accumulate_grad(expanded_grad) 并递归
    Tensor expanded(input_shape_);
    for (float& x : expanded.data_)
        x = grad->data_[0] / n_;
    auto ep = std::make_shared<Tensor>(expanded);
    if (next_[0]->requires_grad_)
        next_[0]->backward(ep);
}

// ============================================================
// ReluBackward
// z = relu(x)  =>  dL/dx[i] = grad[i] * (x[i] > 0 ? 1 : 0)
// ============================================================
void ReluBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: local_grad[i] = (saved_tensors_[0]->data_[i] > 0) ? grad->data_[i] : 0
    //       next_[0]->accumulate_grad(local_grad) 并递归
    Tensor local_grad(grad->shape_);
    for (int i = 0; i < grad->numel(); i++)
        local_grad.data_[i] = saved_tensors_[0]->data_[i] > 0 ? grad->data_[i] : 0.f;
    auto lp = std::make_shared<Tensor>(local_grad);
    if (next_[0]->requires_grad_)
        next_[0]->backward(lp);
}

// ============================================================
// SigmoidBackward
// z = sigmoid(x)  =>  dL/dx = grad * z * (1 - z)
// ============================================================
void SigmoidBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: local_grad[i] = grad[i] * output_[i] * (1 - output_[i])
    //       next_[0]->accumulate_grad(local_grad) 并递归
    Tensor local_grad(grad->shape_);
    for (int i = 0; i < grad->numel(); i++) {
        float z = output_->data_[i];
        local_grad.data_[i] = grad->data_[i] * z * (1.f - z);
    }
    auto lp = std::make_shared<Tensor>(local_grad);
    if (next_[0]->requires_grad_)
        next_[0]->backward(lp);
}

// ============================================================
// TanhBackward
// z = tanh(x)  =>  dL/dx = grad * (1 - z^2)
// ============================================================
void TanhBackward::backward(std::shared_ptr<Tensor> grad) {
    // todo: local_grad[i] = grad[i] * (1 - output_[i] * output_[i])
    //       next_[0]->accumulate_grad(local_grad) 并递归
    Tensor local_grad(grad->shape_);
    for (int i = 0; i < grad->numel(); i++) {
        float z = output_->data_[i];
        local_grad.data_[i] = grad->data_[i] * (1.f - z * z);
    }
    auto lp = std::make_shared<Tensor>(local_grad);
    if (next_[0]->requires_grad_)
        next_[0]->backward(lp);
}
