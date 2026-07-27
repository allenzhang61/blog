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

void copy_params(GPTModel& dst, GPTModel& src) {
    auto dst_params = dst.parameters();
    auto src_params = src.parameters();
    ASSERT_EQ(dst_params.size(), src_params.size()) << "copy params size";
    for (size_t i = 0; i < dst_params.size(); ++i) {
        ASSERT_EQ(dst_params[i]->shape(), src_params[i]->shape()) << "copy params shape";
        dst_params[i]->data() = src_params[i]->data();
    }
}

// Metal 后端不可用时跳过所有用例。
class MetalTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!metal_backend_available()) {
            GTEST_SKIP() << metal_backend_status();
        }
    }
};

} // namespace

TEST_F(MetalTest, Availability) {
    EXPECT_EQ(BackendRegistry::get(Device::parse("metal")).name(), "metalBackend");
}

TEST_F(MetalTest, Elementwise) {
    Device metal_device = Device::parse("metal");
    Tensor a_cpu = Tensor::from_vector({1.0, 2.0, 3.0, 4.0}, {2, 2});
    Tensor b_cpu = Tensor::from_vector({5.0, 6.0, 7.0, 8.0}, {2, 2});
    Tensor a_metal = Tensor::from_vector(a_cpu.data(), a_cpu.shape(), metal_device);
    Tensor b_metal = Tensor::from_vector(b_cpu.data(), b_cpu.shape(), metal_device);
    expect_tensor_close(ops::add(a_metal, b_metal), ops::add(a_cpu, b_cpu), 1e-5, "metal add");
    expect_tensor_close(ops::mul(a_metal, b_metal), ops::mul(a_cpu, b_cpu), 1e-5, "metal mul");
    expect_tensor_close(ops::mul_scalar(a_metal, 0.5), ops::mul_scalar(a_cpu, 0.5), 1e-5, "metal mul_scalar");
}

TEST_F(MetalTest, Matmul) {
    Device metal_device = Device::parse("metal");
    Tensor a_cpu = Tensor::from_vector({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    Tensor b_cpu = Tensor::from_vector({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {3, 2});
    Tensor a_metal = Tensor::from_vector(a_cpu.data(), a_cpu.shape(), metal_device);
    Tensor b_metal = Tensor::from_vector(b_cpu.data(), b_cpu.shape(), metal_device);
    expect_tensor_close(ops::matmul(a_metal, b_metal), ops::matmul(a_cpu, b_cpu), 1e-4, "metal matmul");
}

TEST_F(MetalTest, BatchMatmul) {
    Device metal_device = Device::parse("metal");
    Tensor a_cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 2, 4});
    Tensor b_cpu = Tensor::from_vector({1, 0, 0, 1, 2, 1, 1, 2}, {1, 1, 4, 2});
    Tensor a_metal = Tensor::from_vector(a_cpu.data(), a_cpu.shape(), metal_device);
    Tensor b_metal = Tensor::from_vector(b_cpu.data(), b_cpu.shape(), metal_device);
    expect_tensor_close(ops::batch_matmul(a_metal, b_metal), ops::batch_matmul(a_cpu, b_cpu), 1e-4, "metal batch_matmul");
}

TEST_F(MetalTest, SoftmaxLayernormEmbeddingCrossEntropy) {
    Device metal_device = Device::parse("metal");

    Tensor s_cpu = Tensor::from_vector({1, 2, 3, 1, 0, -1}, {2, 3});
    Tensor s_metal = Tensor::from_vector(s_cpu.data(), s_cpu.shape(), metal_device);
    expect_tensor_close(ops::softmax(s_metal), ops::softmax(s_cpu), 1e-5, "metal softmax");

    Tensor x_cpu = Tensor::from_vector({1, 2, 3, 4, 2, 3, 4, 5}, {2, 4});
    Tensor scale_cpu = Tensor::from_vector({1, 1, 1, 1}, {4});
    Tensor shift_cpu = Tensor::from_vector({0, 0, 0, 0}, {4});
    Tensor x_metal = Tensor::from_vector(x_cpu.data(), x_cpu.shape(), metal_device);
    Tensor scale_metal = Tensor::from_vector(scale_cpu.data(), scale_cpu.shape(), metal_device);
    Tensor shift_metal = Tensor::from_vector(shift_cpu.data(), shift_cpu.shape(), metal_device);
    expect_tensor_close(ops::layernorm(x_metal, scale_metal, shift_metal), ops::layernorm(x_cpu, scale_cpu, shift_cpu),
                        1e-5, "metal layernorm");

    Tensor ids_cpu = Tensor::from_ints({0, 2, 1}, {3});
    Tensor weight_cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor ids_metal = Tensor::from_ints({0, 2, 1}, {3}, metal_device);
    Tensor weight_metal = Tensor::from_vector(weight_cpu.data(), weight_cpu.shape(), metal_device);
    expect_tensor_close(ops::embedding(ids_metal, weight_metal), ops::embedding(ids_cpu, weight_cpu), 1e-5,
                        "metal embedding");

    Tensor logits_cpu = Tensor::from_vector({1, 2, 3, 1, 3, 2}, {1, 2, 3});
    Tensor targets_cpu = Tensor::from_ints({2, 1}, {1, 2});
    Tensor logits_metal = Tensor::from_vector(logits_cpu.data(), logits_cpu.shape(), metal_device);
    Tensor targets_metal = Tensor::from_ints({2, 1}, {1, 2}, metal_device);
    EXPECT_NEAR(ops::cross_entropy(logits_metal, targets_metal).item(),
                ops::cross_entropy(logits_cpu, targets_cpu).item(),
                1e-5) << "metal cross_entropy";
}

TEST_F(MetalTest, Gelu) {
    Device metal_device = Device::parse("metal");
    Tensor x_cpu = Tensor::from_vector({-1.0, 0.0, 1.0, 2.0}, {4});
    Tensor x_metal = Tensor::from_vector(x_cpu.data(), x_cpu.shape(), metal_device);
    expect_tensor_close(ops::gelu(x_metal), ops::gelu(x_cpu), 1e-5, "metal gelu");
}

TEST_F(MetalTest, ModelSubgraph) {
    Device metal_device = Device::parse("metal");
    Tensor x_cpu = Tensor::from_vector({0.5, -1.0, 1.5, 2.0}, {2, 2});
    Tensor w1_cpu = Tensor::from_vector({0.1, 0.2, -0.3, 0.4, 0.5, -0.6}, {2, 3});
    Tensor b1_cpu = Tensor::from_vector({0.01, -0.02, 0.03}, {3});
    Tensor w2_cpu = Tensor::from_vector({0.2, -0.1, 0.3, 0.4, -0.5, 0.6}, {3, 2});

    Tensor x_metal = Tensor::from_vector(x_cpu.data(), x_cpu.shape(), metal_device);
    Tensor w1_metal = Tensor::from_vector(w1_cpu.data(), w1_cpu.shape(), metal_device);
    Tensor b1_metal = Tensor::from_vector(b1_cpu.data(), b1_cpu.shape(), metal_device);
    Tensor w2_metal = Tensor::from_vector(w2_cpu.data(), w2_cpu.shape(), metal_device);

    Tensor cpu = ops::matmul(ops::gelu(ops::add(ops::matmul(x_cpu, w1_cpu), b1_cpu)), w2_cpu);
    Tensor gpu = ops::matmul(ops::gelu(ops::add(ops::matmul(x_metal, w1_metal), b1_metal)), w2_metal);
    expect_tensor_close(gpu, cpu, 1e-4, "metal model subgraph");
}

TEST_F(MetalTest, GptForwardMatchesCpu) {
    GPTConfig cfg;
    cfg.vocab_size = 32;
    cfg.context_length = 4;
    cfg.emb_dim = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;

    GPTModel cpu_model(cfg);
    GPTConfig metal_cfg = cfg;
    metal_cfg.device = Device::parse("metal");
    GPTModel metal_model(metal_cfg);
    copy_params(metal_model, cpu_model);

    Tensor ids_cpu = Tensor::from_ints({1, 2, 3, 4}, {1, 4});
    Tensor ids_metal = Tensor::from_ints({1, 2, 3, 4}, {1, 4}, metal_cfg.device);
    Tensor cpu_logits = cpu_model.forward(ids_cpu);
    Tensor metal_logits = metal_model.forward(ids_metal);
    expect_tensor_close(metal_logits, cpu_logits, 2e-3, "metal gpt forward");
}

TEST_F(MetalTest, GptTrainStep) {
    GPTConfig cfg;
    cfg.vocab_size = 32;
    cfg.context_length = 4;
    cfg.emb_dim = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.device = Device::parse("metal");
    GPTModel model(cfg);
    std::vector<int64_t> ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    DataLoader loader(ids, 1, cfg.context_length, cfg.context_length, false, cfg.device);
    AdamW optim(model.parameters(), 1e-3, 0.01);
    double loss = Trainer::train_steps(model, loader, optim, 1);
    EXPECT_TRUE(std::isfinite(loss) && loss > 0.0) << "metal train loss finite";
}
