#include "llm/llm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

using namespace llm;

namespace {

void expect_tensor_close(const Tensor& got, const Tensor& expected, double tol, const std::string& name) {
    ASSERT_EQ(got.shape(), expected.shape()) << name << " shape";
    for (int64_t i = 0; i < got.numel(); ++i) {
        EXPECT_NEAR(got.data()[i], expected.data()[i], tol) << name << " value " << i;
    }
}

void expect_vector_close(const std::vector<double>& got, const std::vector<double>& expected,
                         double tol, const std::string& name) {
    ASSERT_EQ(got.size(), expected.size()) << name << " size";
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], tol) << name << " value " << i;
    }
}

Tensor cuda_tensor(const Tensor& cpu, bool requires_grad = false) {
    return Tensor::from_vector(cpu.data(), cpu.shape(), Device::parse("cuda"), requires_grad);
}

// CUDA 后端不可用时跳过所有用例。
class CudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_backend_available()) {
            GTEST_SKIP() << cuda_backend_status();
        }
    }
};

} // namespace

TEST_F(CudaTest, Availability) {
    EXPECT_EQ(BackendRegistry::get(Device::parse("cuda")).name(), "cudaBackend");
}

TEST_F(CudaTest, ElementwiseForward) {
    Device cuda_device = Device::parse("cuda");
    Tensor a_cpu = Tensor::from_vector({1.0, 2.0, 3.0, 4.0}, {2, 2});
    Tensor b_cpu = Tensor::from_vector({5.0, 6.0, 7.0, 8.0}, {2, 2});
    Tensor a_cuda = Tensor::from_vector(a_cpu.data(), a_cpu.shape(), cuda_device);
    Tensor b_cuda = Tensor::from_vector(b_cpu.data(), b_cpu.shape(), cuda_device);

    expect_tensor_close(ops::add(a_cuda, b_cuda), ops::add(a_cpu, b_cpu), 1e-5, "cuda add");
    expect_tensor_close(ops::mul(a_cuda, b_cuda), ops::mul(a_cpu, b_cpu), 1e-5, "cuda mul");
    expect_tensor_close(ops::div(b_cuda, a_cuda), ops::div(b_cpu, a_cpu), 1e-5, "cuda div");
    expect_tensor_close(ops::sum(a_cuda), ops::sum(a_cpu), 1e-5, "cuda sum");
    expect_tensor_close(ops::mean(a_cuda), ops::mean(a_cpu), 1e-5, "cuda mean");
}

TEST_F(CudaTest, MatmulForward) {
    Device cuda_device = Device::parse("cuda");
    Tensor a_cpu = Tensor::from_vector({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    Tensor b_cpu = Tensor::from_vector({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {3, 2});
    Tensor a_cuda = Tensor::from_vector(a_cpu.data(), a_cpu.shape(), cuda_device);
    Tensor b_cuda = Tensor::from_vector(b_cpu.data(), b_cpu.shape(), cuda_device);

    expect_tensor_close(ops::matmul(a_cuda, b_cuda), ops::matmul(a_cpu, b_cpu), 1e-4, "cuda matmul");
}

TEST_F(CudaTest, CausalMaskForwardBackward) {
    Tensor s_cpu = Tensor::from_vector({
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    }, {1, 1, 3, 3}, {}, true);
    Tensor masked_cpu = ops::causal_mask(s_cpu, 3);
    Tensor loss_cpu = ops::sum(masked_cpu);
    loss_cpu.backward();

    Tensor s_cuda = Tensor::from_vector(s_cpu.data(), s_cpu.shape(), Device::parse("cuda"), true);
    Tensor masked_cuda = ops::causal_mask(s_cuda, 3);
    Tensor loss_cuda = ops::sum(masked_cuda);
    loss_cuda.backward();

    expect_tensor_close(masked_cuda, masked_cpu, 1e-5, "cuda causal mask forward");
    expect_vector_close(s_cuda.grad(), s_cpu.grad(), 1e-5, "cuda causal mask backward");
}

TEST_F(CudaTest, TrainingForwardOps) {
    Device cuda_device = Device::parse("cuda");
    Tensor logits_cpu = Tensor::from_vector({0.1, 1.2, -0.3, 0.5, -0.7, 2.0}, {1, 2, 3});
    Tensor targets_cpu = Tensor::from_ints({1, 2}, {1, 2});
    Tensor logits_cuda = Tensor::from_vector(logits_cpu.data(), logits_cpu.shape(), cuda_device);
    Tensor targets_cuda = Tensor::from_ints({1, 2}, {1, 2}, cuda_device);
    expect_tensor_close(ops::softmax(logits_cuda, -1), ops::softmax(logits_cpu, -1), 1e-5, "cuda softmax");
    expect_tensor_close(ops::log_softmax(logits_cuda, -1), ops::log_softmax(logits_cpu, -1), 1e-5, "cuda log_softmax");
    expect_tensor_close(ops::cross_entropy(logits_cuda, targets_cuda), ops::cross_entropy(logits_cpu, targets_cpu),
                        1e-5, "cuda cross_entropy");

    Tensor ids_cpu = Tensor::from_ints({0, 2, 1}, {3});
    Tensor weight_cpu = Tensor::from_vector({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, {3, 2});
    Tensor ids_cuda = Tensor::from_ints({0, 2, 1}, {3}, cuda_device);
    Tensor weight_cuda = Tensor::from_vector(weight_cpu.data(), weight_cpu.shape(), cuda_device);
    expect_tensor_close(ops::embedding(ids_cuda, weight_cuda), ops::embedding(ids_cpu, weight_cpu),
                        1e-5, "cuda embedding");

    Tensor x_cpu = Tensor::from_vector({1.0, 2.0, 3.0, 4.0}, {2, 2});
    Tensor scale_cpu = Tensor::from_vector({1.5, 0.5}, {2});
    Tensor shift_cpu = Tensor::from_vector({0.1, -0.2}, {2});
    Tensor x_cuda = Tensor::from_vector(x_cpu.data(), x_cpu.shape(), cuda_device);
    Tensor scale_cuda = Tensor::from_vector(scale_cpu.data(), scale_cpu.shape(), cuda_device);
    Tensor shift_cuda = Tensor::from_vector(shift_cpu.data(), shift_cpu.shape(), cuda_device);
    expect_tensor_close(ops::layernorm(x_cuda, scale_cuda, shift_cuda), ops::layernorm(x_cpu, scale_cpu, shift_cpu),
                        1e-5, "cuda layernorm");
    expect_tensor_close(ops::gelu(x_cuda), ops::gelu(x_cpu), 1e-5, "cuda gelu");
}

TEST_F(CudaTest, ElementwiseBackward) {
    Tensor a_cpu = Tensor::from_vector({1.5, -2.0, 3.0}, {3}, {}, true);
    Tensor b_cpu = Tensor::from_vector({0.5, 4.0, -1.0}, {3}, {}, true);
    Tensor loss_cpu = ops::sum(ops::add(ops::pow(ops::mul(a_cpu, b_cpu), 2.0), ops::div(a_cpu, b_cpu)));
    loss_cpu.backward();

    Tensor a_cuda = cuda_tensor(a_cpu, true);
    Tensor b_cuda = cuda_tensor(b_cpu, true);
    Tensor loss_cuda = ops::sum(ops::add(ops::pow(ops::mul(a_cuda, b_cuda), 2.0), ops::div(a_cuda, b_cuda)));
    loss_cuda.backward();

    expect_vector_close(a_cuda.grad(), a_cpu.grad(), 2e-4, "cuda elementwise backward a");
    expect_vector_close(b_cuda.grad(), b_cpu.grad(), 2e-4, "cuda elementwise backward b");
}

TEST_F(CudaTest, ReductionMatmulBackward) {
    Tensor a_cpu = Tensor::from_vector({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, {}, true);
    Tensor b_cpu = Tensor::from_vector({0.5, -1.0, 1.5, 2.0, -0.5, 0.25}, {3, 2}, {}, true);
    Tensor loss_cpu = ops::mean(ops::matmul(a_cpu, b_cpu));
    loss_cpu.backward();

    Tensor a_cuda = cuda_tensor(a_cpu, true);
    Tensor b_cuda = cuda_tensor(b_cpu, true);
    Tensor loss_cuda = ops::mean(ops::matmul(a_cuda, b_cuda));
    loss_cuda.backward();

    expect_vector_close(a_cuda.grad(), a_cpu.grad(), 2e-4, "cuda matmul backward a");
    expect_vector_close(b_cuda.grad(), b_cpu.grad(), 2e-4, "cuda matmul backward b");
}

TEST_F(CudaTest, TrainingBackwardOps) {
    Tensor logits_cpu = Tensor::from_vector({0.2, 1.0, -0.5, 0.7, -0.3, 1.5}, {1, 2, 3}, {}, true);
    Tensor targets_cpu = Tensor::from_ints({1, 2}, {1, 2});
    Tensor loss_cpu = ops::cross_entropy(logits_cpu, targets_cpu);
    loss_cpu.backward();

    Tensor logits_cuda = Tensor::from_vector(logits_cpu.data(), logits_cpu.shape(), Device::parse("cuda"), true);
    Tensor targets_cuda = Tensor::from_ints({1, 2}, {1, 2}, Device::parse("cuda"));
    Tensor loss_cuda = ops::cross_entropy(logits_cuda, targets_cuda);
    loss_cuda.backward();
    expect_vector_close(logits_cuda.grad(), logits_cpu.grad(), 2e-4, "cuda cross entropy backward");

    Tensor ids_cpu = Tensor::from_ints({0, 2, 0}, {3});
    Tensor weight_cpu = Tensor::from_vector({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, {3, 2}, {}, true);
    Tensor embed_loss_cpu = ops::sum(ops::embedding(ids_cpu, weight_cpu));
    embed_loss_cpu.backward();
    Tensor ids_cuda = Tensor::from_ints({0, 2, 0}, {3}, Device::parse("cuda"));
    Tensor weight_cuda = Tensor::from_vector(weight_cpu.data(), weight_cpu.shape(), Device::parse("cuda"), true);
    Tensor embed_loss_cuda = ops::sum(ops::embedding(ids_cuda, weight_cuda));
    embed_loss_cuda.backward();
    expect_vector_close(weight_cuda.grad(), weight_cpu.grad(), 2e-4, "cuda embedding backward");
}

TEST_F(CudaTest, AdamW) {
    Tensor p_cpu = Tensor::from_vector({1.0, -2.0, 3.0}, {3}, {}, true);
    p_cpu.mutable_grad() = {0.1, -0.2, 0.3};
    p_cpu.mark_grad_host_dirty();
    AdamW opt_cpu({&p_cpu}, 0.01, 0.1);
    opt_cpu.step();

    Tensor p_cuda = Tensor::from_vector({1.0, -2.0, 3.0}, {3}, Device::parse("cuda"), true);
    p_cuda.mutable_grad() = {0.1, -0.2, 0.3};
    p_cuda.mark_grad_host_dirty();
    AdamW opt_cuda({&p_cuda}, 0.01, 0.1);
    opt_cuda.step();

    expect_vector_close(p_cuda.data(), p_cpu.data(), 2e-4, "cuda adamw step");
}

TEST_F(CudaTest, SmallSubgraph) {
    Tensor x = Tensor::from_vector({0.5, -1.0, 1.5, 2.0}, {2, 2}, Device::parse("cuda"));
    Tensor w = Tensor::from_vector({0.2, -0.3, 0.4, 0.1}, {2, 2}, Device::parse("cuda"), true);
    AdamW opt({&w}, 0.001, 0.01);
    Tensor loss = ops::mean(ops::gelu(ops::matmul(x, w)));
    loss.backward();
    opt.step();
    EXPECT_EQ(w.data().size(), 4u) << "cuda small subgraph parameter data";
}
