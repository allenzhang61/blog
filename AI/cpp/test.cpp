#include <gtest/gtest.h>
#include "tensor.h"
#include <cmath>
#include <functional>
#include <random>

// ============================================================
// 数值梯度检验辅助（有限差分）
// ============================================================
// eps=1e-3：float32 在值较大（如 20）时，1e-4 只有约 52 ULP，舍入误差会使梯度偏 ~1%
static Tensor numerical_gradient(std::function<float(Tensor)> f,
                                 const Tensor &x, float eps = 1e-3f) {
    Tensor grad(x.shape_);
    for (int i = 0; i < x.numel(); i++) {
        Tensor xp = x, xm = x;
        xp.data_[i] += eps;
        xm.data_[i] -= eps;
        grad.data_[i] = (f(xp) - f(xm)) / (2.f * eps);
    }
    return grad;
}

static bool tensors_near(const Tensor &a, const Tensor &b, float tol = 1e-2f) {
    if (a.shape_ != b.shape_) return false;
    for (int i = 0; i < a.numel(); i++)
        if (std::fabs(a.data_[i] - b.data_[i]) > tol) return false;
    return true;
}

// ============================================================
// 1. 基础属性
// ============================================================
//done
TEST(TensorBasics, NumelAndNdim) {
    Tensor t({2, 3});
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape_[0], 2);
    EXPECT_EQ(t.shape_[1], 3);
}

//done
TEST(TensorBasics, StridesRowMajor) {
    Tensor t({2, 3});
    // 行优先：strides = {3, 1}
    EXPECT_EQ(t.strides_[0], 3);
    EXPECT_EQ(t.strides_[1], 1);

    Tensor t2({2, 3, 4});
    EXPECT_EQ(t2.strides_[0], 12);
    EXPECT_EQ(t2.strides_[1], 4);
    EXPECT_EQ(t2.strides_[2], 1);
}

//done
TEST(TensorBasics, FactoryZeros) {
    auto z = Tensor::zeros({2, 3});
    EXPECT_EQ(z.numel(), 6);
    for (int i = 0; i < z.numel(); i++) {
        EXPECT_FLOAT_EQ(z.data_[i], 0.f);
    }
    // EXPECT_FLOAT_EQ(z.at({0, 0}), 0.f);
    // EXPECT_FLOAT_EQ(z.at({0, 1}), 0.f);
    // EXPECT_FLOAT_EQ(z.at({1, 0}), 0.f);
    // EXPECT_FLOAT_EQ(z.at({1, 1}), 0.f);
}

//done
TEST(TensorBasics, FactoryOnes) {
    auto z = Tensor::ones({2, 3});
    EXPECT_EQ(z.numel(), 6);
    for (int i = 0; i < z.numel(); i++) {
        EXPECT_FLOAT_EQ(z.data_[i], 1.f);
    }
    // auto o = Tensor::ones({3});
    // EXPECT_FLOAT_EQ(o.at({0}), 1.f);
    // EXPECT_FLOAT_EQ(o.at({2}), 1.f);
}

//done
TEST(TensorBasics, FactoryRandn) {
    // randn 实现后，验证均值约为 0，标准差约为 1
    auto r = Tensor::randn({1000});
    float mean = 0.f;
    for (float v: r.data_) mean += v;
    mean /= r.numel();
    EXPECT_NEAR(mean, 0.f, 0.1f);
}

// ============================================================
// 2. 元素访问 / flat_index
// ============================================================
//done
TEST(Indexing, ReadElements) {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.f);
}

//done
TEST(Indexing, WriteElements) {
    Tensor t({2, 3});
    t.at({1, 1}) = 42.f;
    EXPECT_FLOAT_EQ(t.at({1, 1}), 42.f);
}

//done
TEST(Indexing, OutOfBoundsThrows) {
    // flat_index 越界检查实现后，取消注释
    Tensor t({2, 3});
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
}

// ============================================================
// 3. Reshape
// ============================================================
//done
TEST(Reshape, CorrectShape) {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    auto r = t.reshape({3, 2});
    EXPECT_EQ(r.shape_[0], 3);
    EXPECT_EQ(r.shape_[1], 2);
    EXPECT_EQ(r.numel(), 6);
}

//done
TEST(Reshape, DataOrderPreserved) {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    auto r = t.reshape({3, 2});
    EXPECT_FLOAT_EQ(r.at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(r.at({2, 1}), 6.f);
}

//done
TEST(Reshape, InvalidShapeThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.reshape({2, 4}), std::invalid_argument);
}

// ============================================================
// 4. Transpose
// ============================================================
//done
TEST(Transpose, TwoByTwo) {
    // [[1,2],[3,4]] 转置 => [[1,3],[2,4]]
    Tensor t({1, 2, 3, 4}, {2, 2});
    auto tr = t.transpose();
    EXPECT_FLOAT_EQ(tr.at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(tr.at({0, 1}), 3.f);
    EXPECT_FLOAT_EQ(tr.at({1, 0}), 2.f);
    EXPECT_FLOAT_EQ(tr.at({1, 1}), 4.f);
}

//done
TEST(Transpose, ShapeSwapped) {
    Tensor t({2, 3});
    auto tr = t.transpose();
    EXPECT_EQ(tr.shape_[0], 3);
    EXPECT_EQ(tr.shape_[1], 2);
}

// ============================================================
// 5. Element-wise 运算
// ============================================================
//done
TEST(ElementwiseOps, Add) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto c = a + b;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 5.f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 5.f);
}

//done
TEST(ElementwiseOps, Sub) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto d = a - b;
    EXPECT_FLOAT_EQ(d.at({0, 0}), -3.f);
    EXPECT_FLOAT_EQ(d.at({1, 1}), 3.f);
}

//done
TEST(ElementwiseOps, Mul) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto e = a * b;
    EXPECT_FLOAT_EQ(e.at({0, 0}), 4.f);
    EXPECT_FLOAT_EQ(e.at({0, 1}), 6.f);
}

//done
TEST(ElementwiseOps, Div) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto f = a / b;
    EXPECT_NEAR(f.at({0, 0}), 0.25f, 1e-5f);
    EXPECT_NEAR(f.at({1, 1}), 4.f, 1e-5f);
}

//done
TEST(ElementwiseOps, ShapeMismatchThrows) {
    Tensor a({2, 3});
    Tensor b({3, 2});
    EXPECT_THROW(a + b, std::invalid_argument);
}

// ============================================================
// 6. Matmul
// ============================================================
//done
TEST(Matmul, VectorDot) {
    // [1,2] @ [[5],[6]] = [[17]]
    Tensor a({1, 2}, {1, 2});
    Tensor b({5, 6}, {2, 1});
    auto c = a.matmul(b);
    EXPECT_EQ(c.shape_[0], 1);
    EXPECT_EQ(c.shape_[1], 1);
    EXPECT_NEAR(c.at({0, 0}), 17.f, 1e-5f);
}

//done
TEST(Matmul, IdentityMatrix) {
    Tensor eye({1, 0, 0, 1}, {2, 2});
    Tensor m({3, 4, 5, 6}, {2, 2});
    auto r = eye.matmul(m);
    EXPECT_NEAR(r.at({0, 0}), 3.f, 1e-5f);
    EXPECT_NEAR(r.at({0, 1}), 4.f, 1e-5f);
    EXPECT_NEAR(r.at({1, 0}), 5.f, 1e-5f);
    EXPECT_NEAR(r.at({1, 1}), 6.f, 1e-5f);
}

//done
TEST(Matmul, ShapeMismatchThrows) {
    // matmul 合法性检查实现后，取消注释
    Tensor a({2, 3});
    Tensor b({2, 3});
    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

// ============================================================
// 7. 归约
// ============================================================
//done
TEST(Reduction, SumGlobal) {
    Tensor t({1, 2, 3, 4}, {2, 2});
    auto s = t.sum();
    EXPECT_NEAR(s.data_[0], 10.f, 1e-5f);
}

//done
TEST(Reduction, MeanGlobal) {
    Tensor t({1, 2, 3, 4}, {2, 2});
    auto m = t.mean();
    EXPECT_NEAR(m.data_[0], 2.5f, 1e-5f);
}

// ============================================================
// 8. 激活函数
// ============================================================
//done
TEST(Activations, Relu) {
    Tensor t(std::vector<float>{-2.f, -1.f, 0.f, 1.f, 2.f}, {5});
    auto r = t.relu();
    EXPECT_FLOAT_EQ(r.at({0}), 0.f);
    EXPECT_FLOAT_EQ(r.at({1}), 0.f);
    EXPECT_FLOAT_EQ(r.at({2}), 0.f);
    EXPECT_FLOAT_EQ(r.at({3}), 1.f);
    EXPECT_FLOAT_EQ(r.at({4}), 2.f);
}

//done
TEST(Activations, SigmoidAtZero) {
    Tensor t(std::vector<float>{0.f}, {1});
    EXPECT_NEAR(t.sigmoid().at({0}), 0.5f, 1e-5f);
}

//done
TEST(Activations, SigmoidAtTwo) {
    Tensor t(std::vector<float>{2.f}, {1});
    EXPECT_NEAR(t.sigmoid().at({0}), 0.8808f, 1e-3f);
}

//done
TEST(Activations, TanhAtZero) {
    Tensor t(std::vector<float>{0.f}, {1});
    EXPECT_NEAR(t.tanh_().at({0}), 0.f, 1e-5f);
}

//done
TEST(Activations, TanhAtOne) {
    Tensor t(std::vector<float>{1.f}, {1});
    EXPECT_NEAR(t.tanh_().at({0}), 0.7616f, 1e-3f);
}

// ============================================================
// 9. 计算图构造
// ============================================================
//done
TEST(GradGraph, NonLeafHasGradFn) {
    auto a = Tensor::ones({2, 2}, true);
    auto b = Tensor::ones({2, 2}, true);
    auto c = a + b;
    EXPECT_TRUE(c.requires_grad_);
    EXPECT_FALSE(c.is_leaf_);
    EXPECT_NE(c.grad_fn_, nullptr);
}

//done
TEST(GradGraph, LeafHasNoGradFn) {
    auto a = Tensor::ones({2, 2}, true);
    EXPECT_TRUE(a.is_leaf_);
    EXPECT_EQ(a.grad_fn_, nullptr);
}

//done
TEST(GradGraph, NoGradPropagates) {
    // 两个 requires_grad=false 的 Tensor 相加，结果也不需要梯度
    auto a = Tensor::ones({2, 2}, false);
    auto b = Tensor::ones({2, 2}, false);
    auto c = a + b;
    EXPECT_FALSE(c.requires_grad_);
}

// ============================================================
// 10. 数值梯度检验
// ============================================================

TEST(NumericalGrad, Add) {
    Tensor a({1, 2, 3, 4}, {2, 2}, true);
    Tensor b({4, 3, 2, 1}, {2, 2}, true);
    auto num = numerical_gradient([&](Tensor x) { return (x + b).sum().data_[0]; }, a);

    auto out = (a + b).sum();
    out.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_TRUE(tensors_near(*a.grad_, num));
}

TEST(NumericalGrad, Mul) {
    Tensor a({1, 2, 3, 4}, {2, 2}, true);
    Tensor b({4, 3, 2, 1}, {2, 2}, true);
    auto num = numerical_gradient([&](Tensor x) { return (x * b).sum().data_[0]; }, a);

    auto out = (a * b).sum();
    out.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_TRUE(tensors_near(*a.grad_, num));
}

TEST(NumericalGrad, Relu) {
    Tensor a({-1, 0.5f, 2, -0.3f}, {4}, true);
    auto num = numerical_gradient([&](Tensor x) { return x.relu().sum().data_[0]; }, a);

    auto out = a.relu().sum();
    out.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_TRUE(tensors_near(*a.grad_, num));
}

TEST(NumericalGrad, Sigmoid) {
    Tensor a({-1, 0, 1, 2}, {4}, true);
    auto num = numerical_gradient([&](Tensor x) { return x.sigmoid().sum().data_[0]; }, a);

    auto out = a.sigmoid().sum();
    out.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_TRUE(tensors_near(*a.grad_, num));
}

TEST(NumericalGrad, Tanh) {
    Tensor a({-1, 0, 1, 2}, {4}, true);
    auto num = numerical_gradient([&](Tensor x) { return x.tanh_().sum().data_[0]; }, a);

    auto out = a.tanh_().sum();
    out.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_TRUE(tensors_near(*a.grad_, num));
}

TEST(NumericalGrad, Matmul) {
    Tensor a({1, 2, 3, 4, 5, 6}, {2, 3}, true);
    Tensor b({1, 0, 0, 1, 0, 0}, {3, 2}, true);
    auto num = numerical_gradient([&](Tensor x) { return x.matmul(b).sum().data_[0]; }, a);

    auto out = a.matmul(b).sum();
    out.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_TRUE(tensors_near(*a.grad_, num));
}

TEST(NumericalGrad, ChainSigmoidMatmul) {
    // todo: 链式运算 sigmoid(a @ b).sum() 的梯度检验
    // GTEST_SKIP() << "需先实现 matmul + sigmoid backward";

    // sigmoid(a @ b).sum() 对 a 的梯度检验
    Tensor a(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, {2, 3}, true);
    Tensor b(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, {3, 2}, true);

    auto num_a = numerical_gradient(
        [&](Tensor x) { return x.matmul(b).sigmoid().sum().data_[0]; }, a);

    auto out = a.matmul(b).sigmoid().sum();
    out.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_TRUE(tensors_near(*a.grad_, num_a));
}

// ============================================================
// 11. 链式反向传播
// ============================================================
TEST(ChainBackward, ReluMul) {
    // loss = sum(relu(a * b))
    // a[0]=1, b[0]=2 => a*b=2>0  => grad_a[0]=b[0]=2
    // a[1]=-2,b[1]=2 => a*b=-4<0 => grad_a[1]=0（relu 截断）
    Tensor a({1, -2, 3, -4}, {4}, true);
    Tensor b({2, 2, 2, 2}, {4}, true);

    auto loss = (a * b).relu().sum();
    loss.backward();

    ASSERT_NE(a.grad_, nullptr);
    EXPECT_NEAR(a.grad_->at({0}), 2.f, 1e-5f);
    EXPECT_NEAR(a.grad_->at({1}), 0.f, 1e-5f);
    EXPECT_NEAR(a.grad_->at({2}), 2.f, 1e-5f);
    EXPECT_NEAR(a.grad_->at({3}), 0.f, 1e-5f);
}

// ============================================================
// 12. 线性回归
// ============================================================
TEST(LinearRegression, RecoverWeights) {
    // 真实关系：y = 2x + 3
    // 偏置技巧：X = [x, 1]，w_param = [w, b]^T
    //           y_pred = X @ w_param，避免 broadcasting
    // loss = mean((y_pred - y_true)^2)，全批梯度下降
    const int N = 50;
    const float true_w = 2.f, true_b = 3.f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> x_dist(-1.f, 1.f);
    std::normal_distribution<float> noise(0.f, 0.1f);  // σ=0.1 的高斯噪声

    Tensor X({N, 2});
    Tensor y_true({N, 1});
    for (int i = 0; i < N; i++) {
        float xi = x_dist(rng);
        X.at({i, 0}) = xi;
        X.at({i, 1}) = 1.f;
        y_true.at({i, 0}) = true_w * xi + true_b + noise(rng);
    }

    Tensor w_param({2, 1}, true);  // [w, b]，初始化为零

    const float lr = 0.1f;
    for (int epoch = 0; epoch < 300; epoch++) {
        auto y_pred = X.matmul(w_param);
        auto diff   = y_pred - y_true;
        auto loss   = (diff * diff).mean();

        w_param.zero_grad();
        loss.backward();

        for (int i = 0; i < w_param.numel(); i++)
            w_param.data_[i] -= lr * w_param.grad_->data_[i];

        if (epoch % 50 == 0)
            std::printf("epoch %3d  loss=%.6f\n", epoch, loss.data_[0]);
    }

    EXPECT_NEAR(w_param.at({0, 0}), true_w, 0.1f);
    EXPECT_NEAR(w_param.at({1, 0}), true_b, 0.1f);
}

// ============================================================
// 13. zero_grad
// ============================================================
TEST(ZeroGrad, ClearsGradient) {
    Tensor a({1, 2, 3, 4}, {4}, true);
    a.sum().backward();

    ASSERT_NE(a.grad_, nullptr);
    a.zero_grad();
    for (int i = 0; i < a.grad_->numel(); i++)
        EXPECT_FLOAT_EQ(a.grad_->data_[i], 0.f);
}
