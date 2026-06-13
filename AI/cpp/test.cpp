#include <gtest/gtest.h>
#include "tensor.h"
#include <cmath>
#include <functional>

// ============================================================
// 数值梯度检验辅助（有限差分）
// ============================================================
static Tensor numerical_gradient(std::function<float(Tensor)> f,
                                 const Tensor &x, float eps = 1e-4f) {
    Tensor grad(x.shape_);
    for (int i = 0; i < x.numel(); i++) {
        Tensor xp = x, xm = x;
        xp.data_[i] += eps;
        xm.data_[i] -= eps;
        grad.data_[i] = (f(xp) - f(xm)) / (2.f * eps);
    }
    return grad;
}

static bool tensors_near(const Tensor &a, const Tensor &b, float tol = 1e-3f) {
    if (a.shape_ != b.shape_) return false;
    for (int i = 0; i < a.numel(); i++)
        if (std::fabs(a.data_[i] - b.data_[i]) > tol) return false;
    return true;
}

// ============================================================
// 1. 基础属性
// ============================================================
TEST(TensorBasics, NumelAndNdim) {
    //done
    Tensor t({2, 3});
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape_[0], 2);
    EXPECT_EQ(t.shape_[1], 3);
}

TEST(TensorBasics, StridesRowMajor) {
    Tensor t({2, 3});
    // 行优先：strides = {3, 1}
    EXPECT_EQ(t.strides_[0], 3);
    EXPECT_EQ(t.strides_[1], 1);
}

TEST(TensorBasics, FactoryZeros) {
    auto z = Tensor::zeros({2, 2});
    EXPECT_FLOAT_EQ(z.at({0, 0}), 0.f);
    EXPECT_FLOAT_EQ(z.at({1, 1}), 0.f);
}

TEST(TensorBasics, FactoryOnes) {
    auto o = Tensor::ones({3});
    EXPECT_FLOAT_EQ(o.at({0}), 1.f);
    EXPECT_FLOAT_EQ(o.at({2}), 1.f);
}

TEST(TensorBasics, FactoryRandn) {
    // todo: randn 实现后，验证均值约为 0，标准差约为 1
    auto r = Tensor::randn({1000});
    float mean = 0.f;
    for (float v: r.data_) mean += v;
    mean /= r.numel();
    EXPECT_NEAR(mean, 0.f, 0.1f);
}

// ============================================================
// 2. 元素访问 / flat_index
// ============================================================
TEST(Indexing, ReadElements) {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.f);
}

TEST(Indexing, WriteElements) {
    Tensor t({2, 3});
    t.at({1, 1}) = 42.f;
    EXPECT_FLOAT_EQ(t.at({1, 1}), 42.f);
}

TEST(Indexing, OutOfBoundsThrows) {
    // todo: flat_index 越界检查实现后，取消注释
    // Tensor t({2, 3});
    // EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    GTEST_SKIP() << "需先实现 flat_index 越界检查";
}

// ============================================================
// 3. Reshape
// ============================================================
TEST(Reshape, CorrectShape) {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    auto r = t.reshape({3, 2});
    EXPECT_EQ(r.shape_[0], 3);
    EXPECT_EQ(r.shape_[1], 2);
    EXPECT_EQ(r.numel(), 6);
}

TEST(Reshape, DataOrderPreserved) {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    auto r = t.reshape({3, 2});
    EXPECT_FLOAT_EQ(r.at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(r.at({2, 1}), 6.f);
}

TEST(Reshape, InvalidShapeThrows) {
    // todo: reshape 合法性检查实现后，取消注释
    // Tensor t({2, 3});
    // EXPECT_THROW(t.reshape({2, 4}), std::invalid_argument);
    GTEST_SKIP() << "需先实现 reshape 合法性检查";
}

// ============================================================
// 4. Transpose
// ============================================================
TEST(Transpose, TwoByTwo) {
    // [[1,2],[3,4]] 转置 => [[1,3],[2,4]]
    Tensor t({1, 2, 3, 4}, {2, 2});
    auto tr = t.transpose();
    EXPECT_FLOAT_EQ(tr.at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(tr.at({0, 1}), 3.f);
    EXPECT_FLOAT_EQ(tr.at({1, 0}), 2.f);
    EXPECT_FLOAT_EQ(tr.at({1, 1}), 4.f);
}

TEST(Transpose, ShapeSwapped) {
    Tensor t({2, 3});
    auto tr = t.transpose();
    EXPECT_EQ(tr.shape_[0], 3);
    EXPECT_EQ(tr.shape_[1], 2);
}

// ============================================================
// 5. Element-wise 运算
// ============================================================
TEST(ElementwiseOps, Add) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto c = a + b;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 5.f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 5.f);
}

TEST(ElementwiseOps, Sub) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto d = a - b;
    EXPECT_FLOAT_EQ(d.at({0, 0}), -3.f);
    EXPECT_FLOAT_EQ(d.at({1, 1}), 3.f);
}

TEST(ElementwiseOps, Mul) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto e = a * b;
    EXPECT_FLOAT_EQ(e.at({0, 0}), 4.f);
    EXPECT_FLOAT_EQ(e.at({0, 1}), 6.f);
}

TEST(ElementwiseOps, Div) {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({4, 3, 2, 1}, {2, 2});
    auto f = a / b;
    EXPECT_NEAR(f.at({0, 0}), 0.25f, 1e-5f);
    EXPECT_NEAR(f.at({1, 1}), 4.f, 1e-5f);
}

TEST(ElementwiseOps, ShapeMismatchThrows) {
    Tensor a({2, 3});
    Tensor b({3, 2});
    EXPECT_THROW(a + b, std::invalid_argument);
}

// ============================================================
// 6. Matmul
// ============================================================
TEST(Matmul, VectorDot) {
    // [1,2] @ [[5],[6]] = [[17]]
    Tensor a({1, 2}, {1, 2});
    Tensor b({5, 6}, {2, 1});
    auto c = a.matmul(b);
    EXPECT_EQ(c.shape_[0], 1);
    EXPECT_EQ(c.shape_[1], 1);
    EXPECT_NEAR(c.at({0, 0}), 17.f, 1e-5f);
}

TEST(Matmul, IdentityMatrix) {
    Tensor eye({1, 0, 0, 1}, {2, 2});
    Tensor m({3, 4, 5, 6}, {2, 2});
    auto r = eye.matmul(m);
    EXPECT_NEAR(r.at({0, 0}), 3.f, 1e-5f);
    EXPECT_NEAR(r.at({0, 1}), 4.f, 1e-5f);
    EXPECT_NEAR(r.at({1, 0}), 5.f, 1e-5f);
    EXPECT_NEAR(r.at({1, 1}), 6.f, 1e-5f);
}

TEST(Matmul, ShapeMismatchThrows) {
    // todo: matmul 合法性检查实现后，取消注释
    // Tensor a({2, 3});
    // Tensor b({2, 3});
    // EXPECT_THROW(a.matmul(b), std::invalid_argument);
    GTEST_SKIP() << "需先实现 matmul 合法性检查";
}

// ============================================================
// 7. 归约
// ============================================================
TEST(Reduction, SumGlobal) {
    Tensor t({1, 2, 3, 4}, {2, 2});
    auto s = t.sum();
    EXPECT_NEAR(s.data_[0], 10.f, 1e-5f);
}

TEST(Reduction, MeanGlobal) {
    Tensor t({1, 2, 3, 4}, {2, 2});
    auto m = t.mean();
    EXPECT_NEAR(m.data_[0], 2.5f, 1e-5f);
}

// ============================================================
// 8. 激活函数
// ============================================================
TEST(Activations, Relu) {
    Tensor t(std::vector<float>{-2.f, -1.f, 0.f, 1.f, 2.f}, {5});
    auto r = t.relu();
    EXPECT_FLOAT_EQ(r.at({0}), 0.f);
    EXPECT_FLOAT_EQ(r.at({1}), 0.f);
    EXPECT_FLOAT_EQ(r.at({2}), 0.f);
    EXPECT_FLOAT_EQ(r.at({3}), 1.f);
    EXPECT_FLOAT_EQ(r.at({4}), 2.f);
}

TEST(Activations, SigmoidAtZero) {
    Tensor t(std::vector<float>{0.f}, {1});
    EXPECT_NEAR(t.sigmoid().at({0}), 0.5f, 1e-5f);
}

TEST(Activations, SigmoidAtTwo) {
    Tensor t(std::vector<float>{2.f}, {1});
    EXPECT_NEAR(t.sigmoid().at({0}), 0.8808f, 1e-3f);
}

TEST(Activations, TanhAtZero) {
    Tensor t(std::vector<float>{0.f}, {1});
    EXPECT_NEAR(t.tanh_().at({0}), 0.f, 1e-5f);
}

TEST(Activations, TanhAtOne) {
    Tensor t(std::vector<float>{1.f}, {1});
    EXPECT_NEAR(t.tanh_().at({0}), 0.7616f, 1e-3f);
}

// ============================================================
// 9. 计算图构造
// ============================================================
TEST(GradGraph, NonLeafHasGradFn) {
    auto a = Tensor::ones({2, 2}, true);
    auto b = Tensor::ones({2, 2}, true);
    auto c = a + b;
    EXPECT_TRUE(c.requires_grad_);
    EXPECT_FALSE(c.is_leaf_);
    EXPECT_NE(c.grad_fn_, nullptr);
}

TEST(GradGraph, LeafHasNoGradFn) {
    auto a = Tensor::ones({2, 2}, true);
    EXPECT_TRUE(a.is_leaf_);
    EXPECT_EQ(a.grad_fn_, nullptr);
}

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
    GTEST_SKIP() << "需先实现 matmul + sigmoid backward";
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
// 12. zero_grad
// ============================================================
TEST(ZeroGrad, ClearsGradient) {
    Tensor a({1, 2, 3, 4}, {4}, true);
    a.sum().backward();

    ASSERT_NE(a.grad_, nullptr);
    a.zero_grad();
    for (int i = 0; i < a.grad_->numel(); i++)
        EXPECT_FLOAT_EQ(a.grad_->data_[i], 0.f);
}
