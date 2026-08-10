#include <gtest/gtest.h>
#include "tensor.h"
#include <cmath>
#include <functional>
#include <random>
#include <fstream>
#include <algorithm>
#include <chrono>

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

// ============================================================
// MNIST IDX 格式读取辅助
// ============================================================
static std::vector<uint8_t> read_idx_images(const std::string &path, int &n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    auto read32 = [&]() {
        uint8_t b[4]; f.read((char *)b, 4);
        return (int)((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);
    };
    read32();           // magic
    n = read32();
    int rows = read32(), cols = read32();
    std::vector<uint8_t> data(n * rows * cols);
    f.read((char *)data.data(), data.size());
    return data;
}

static std::vector<uint8_t> read_idx_labels(const std::string &path, int &n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    auto read32 = [&]() {
        uint8_t b[4]; f.read((char *)b, 4);
        return (int)((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);
    };
    read32();           // magic
    n = read32();
    std::vector<uint8_t> labels(n);
    f.read((char *)labels.data(), labels.size());
    return labels;
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
    auto num = numerical_gradient([&](const Tensor& x) { return (x + b).sum().data_[0]; }, a);

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
    auto num = numerical_gradient([&](const Tensor& x) { return x.tanh_().sum().data_[0]; }, a);

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
// 13. 多层感知机 MNIST 识别
// ============================================================
TEST(MLP, MnistDigitRecognition) {
    const std::string dir =
        "/Users/zyl/codes/mygithub/blog/AI/chapter-04/mnist/";
    int n_tr_img, n_tr_lbl, n_te_img, n_te_lbl;
    auto tr_img = read_idx_images(dir + "train-images-idx3-ubyte", n_tr_img);
    auto tr_lbl = read_idx_labels(dir + "train-labels-idx1-ubyte", n_tr_lbl);
    auto te_img = read_idx_images(dir + "t10k-images-idx3-ubyte",  n_te_img);
    auto te_lbl = read_idx_labels(dir + "t10k-labels-idx1-ubyte",  n_te_lbl);
    if (tr_img.empty() || te_img.empty())
        GTEST_SKIP() << "MNIST 数据文件未找到：" << dir;

    // 超参数
    const int D = 784, H = 128, C = 10;
    const int N_TRAIN = 5000, BATCH = 32, EPOCHS = 10;
    const float LR = 0.05f;

    // He 初始化（针对 ReLU）
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);

    Tensor W1({D, H}, true);
    float s1 = std::sqrt(2.f / D);
    for (float &v : W1.data_) v = nd(rng) * s1;

    Tensor W2({H, C}, true);
    float s2 = std::sqrt(2.f / H);
    for (float &v : W2.data_) v = nd(rng) * s2;

    // 训练：784 --[W1]--> relu --> 128 --[W2]--> sigmoid --> 10
    // 损失：MSE（与 one-hot 标签比较）
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.f;
        int n_batches = 0;

        for (int start = 0; start < N_TRAIN; start += BATCH) {
            int bs = std::min(BATCH, N_TRAIN - start);

            // 构建 mini-batch
            Tensor X_b({bs, D});
            Tensor y_b({bs, C});   // one-hot，初始为全零
            for (int i = 0; i < bs; i++) {
                int idx = start + i;
                for (int j = 0; j < D; j++)
                    X_b.data_[i * D + j] = tr_img[idx * D + j] / 255.f;
                y_b.data_[i * C + (int)tr_lbl[idx]] = 1.f;
            }

            // 前向
            auto h    = X_b.matmul(W1).relu();
            auto out  = h.matmul(W2).sigmoid();
            auto diff = out - y_b;
            auto loss = (diff * diff).mean();

            total_loss += loss.data_[0];
            n_batches++;

            // 反向 + SGD
            W1.zero_grad();
            W2.zero_grad();
            loss.backward();

            for (int i = 0; i < W1.numel(); i++) W1.data_[i] -= LR * W1.grad_->data_[i];
            for (int i = 0; i < W2.numel(); i++) W2.data_[i] -= LR * W2.grad_->data_[i];
        }

        std::printf("epoch %2d  loss=%.4f\n", epoch + 1, total_loss / n_batches);
    }

    // 在前 1000 个测试样本上评估准确率
    int correct = 0;
    const int N_TEST = 1000;
    for (int i = 0; i < N_TEST; i++) {
        Tensor x({1, D});
        for (int j = 0; j < D; j++)
            x.data_[j] = te_img[i * D + j] / 255.f;

        auto out = x.matmul(W1).relu().matmul(W2);

        int pred = 0;
        for (int j = 1; j < C; j++)
            if (out.data_[j] > out.data_[pred]) pred = j;

        if (pred == (int)te_lbl[i]) correct++;
    }

    float acc = (float)correct / N_TEST;
    std::printf("test accuracy = %.1f%%  (%d/%d)\n", acc * 100.f, correct, N_TEST);
    EXPECT_GT(acc, 0.70f);
}

// ============================================================
// 14. matmul 访问方式性能对比
// ============================================================
TEST(Benchmark, MatmulAtVsStride) {
    // 模拟 MLP 第一层：(32, 784) @ (784, 128)
    Tensor A({32, 784}), B({784, 128}), C({32, 128});
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (float &v : A.data_) v = d(rng);
    for (float &v : B.data_) v = d(rng);

    const int REPEAT = 20;

    // --- at() 版 ---
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPEAT; r++) {
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 128; j++) {
                float s = 0;
                for (int k = 0; k < 784; k++)
                    s += A.at({i, k}) * B.at({k, j});
                C.at({i, j}) = s;
            }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // --- 直接 strides 版 ---
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPEAT; r++) {
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 128; j++) {
                float s = 0;
                for (int k = 0; k < 784; k++)
                    s += A.data_[i * A.strides_[0] + k * A.strides_[1]]
                       * B.data_[k * B.strides_[0] + j * B.strides_[1]];
                C.data_[i * C.strides_[0] + j * C.strides_[1]] = s;
            }
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    auto ms_at     = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto ms_stride = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("at() x%d:     %.1f ms  (%.2f ms/matmul)\n", REPEAT, ms_at,     ms_at     / REPEAT);
    std::printf("stride x%d:   %.1f ms  (%.2f ms/matmul)\n", REPEAT, ms_stride, ms_stride / REPEAT);
    std::printf("speedup:      %.1fx\n", ms_at / ms_stride);
}

// ============================================================
// 15. zero_grad
// ============================================================
TEST(ZeroGrad, ClearsGradient) {
    Tensor a({1, 2, 3, 4}, {4}, true);
    a.sum().backward();

    ASSERT_NE(a.grad_, nullptr);
    a.zero_grad();
    for (int i = 0; i < a.grad_->numel(); i++)
        EXPECT_FLOAT_EQ(a.grad_->data_[i], 0.f);
}
