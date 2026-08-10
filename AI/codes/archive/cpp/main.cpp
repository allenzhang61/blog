#include <iostream>
#include "tensor.h"

int main() {
    // ---- 基础构造 ----
    auto a = Tensor::randn({2, 3}, /*requires_grad=*/true);
    auto b = Tensor::ones ({2, 3}, /*requires_grad=*/true);
    a.print("a");
    b.print("b");

    // ---- element-wise 运算 ----
    auto c = a + b;
    c.print("a + b");

    auto d = a * b;
    d.print("a * b");

    // ---- matmul ----
    // (2,3) @ (3,2) => (2,2)
    auto w  = Tensor::randn({3, 2}, true);
    auto out = a.matmul(w);
    out.print("a @ w");

    // ---- 激活函数 ----
    auto r = a.relu();
    r.print("relu(a)");

    auto s = a.sigmoid();
    s.print("sigmoid(a)");

    // ---- 求和并反向传播 ----
    auto loss = out.sum();
    loss.print("loss");
    loss.backward();

    // 查看梯度
    if (a.grad_) a.grad_->print("grad of a");
    if (w.grad_) w.grad_->print("grad of w");

    return 0;
}
