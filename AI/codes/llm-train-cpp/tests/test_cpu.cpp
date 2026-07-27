#include "llm/llm.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace llm;

TEST(TensorBasics, ShapeDtypeDevice) {
    Tensor x = Tensor::ones({2, 3});
    EXPECT_EQ(x.numel(), 6);
    EXPECT_EQ(x.shape()[0], 2);
    EXPECT_EQ(x.shape()[1], 3);
    EXPECT_EQ(to_string(x.dtype()), "float32");
    EXPECT_EQ(x.device().type, DeviceType::CPU);
}

TEST(BackendPlaceholders, RegistryAndSelect) {
    EXPECT_EQ(BackendRegistry::get(Device::parse("cpu")).name(), "CPUBackend");
    EXPECT_EQ(select_device_from_arg_or_env("").type, DeviceType::CPU);
    EXPECT_EQ(select_device_from_arg_or_env("metal").type, DeviceType::Metal);

    bool cuda_failed = false;
    try {
        BackendRegistry::get(Device::parse("cuda"));
    } catch (const std::runtime_error& err) {
        cuda_failed = std::string(err.what()).find("CUDA backend is unavailable") != std::string::npos ||
                      std::string(err.what()).find("CUDA backend compiled") != std::string::npos;
    }
    if (!cuda_backend_available()) EXPECT_TRUE(cuda_failed) << "cuda unavailable path";

    bool metal_failed = false;
    try {
        BackendRegistry::get(Device::parse("metal"));
    } catch (const std::runtime_error& err) {
        metal_failed = std::string(err.what()).find("Metal backend is unavailable") != std::string::npos;
    }
    if (!metal_backend_available()) EXPECT_TRUE(metal_failed) << "metal unavailable path";
}

TEST(OpsAndAutograd, MatmulBackward) {
    Tensor a = Tensor::from_vector({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor b = Tensor::from_vector({5, 6, 7, 8}, {2, 2}, {}, true);
    Tensor c = ops::matmul(a, b);
    EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 2}));
    EXPECT_NEAR(c.data()[0], 19.0, 1e-9);
    EXPECT_NEAR(c.data()[3], 50.0, 1e-9);

    Tensor loss = ops::sum(c);
    loss.backward();
    EXPECT_EQ(a.grad().size(), a.data().size());
    EXPECT_EQ(b.grad().size(), b.data().size());
    EXPECT_NEAR(a.grad()[0], 11.0, 1e-9);
    EXPECT_NEAR(b.grad()[0], 4.0, 1e-9);
}

TEST(NeuralOps, CrossEntropyAndSoftmax) {
    Tensor logits = Tensor::from_vector({1, 2, 3, 1, 3, 2}, {1, 2, 3}, {}, true);
    Tensor targets = Tensor::from_ints({2, 1}, {1, 2});
    Tensor loss = ops::cross_entropy(logits, targets);
    EXPECT_GT(loss.item(), 0.0);
    loss.backward();
    EXPECT_EQ(logits.grad().size(), logits.data().size());

    Tensor x = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    Tensor s = ops::softmax(x, -1);
    EXPECT_NEAR(s.data()[0] + s.data()[1], 1.0, 1e-9);
    EXPECT_NEAR(s.data()[2] + s.data()[3], 1.0, 1e-9);
}

TEST(BatchMatmul, Backward) {
    Tensor a = Tensor::from_vector({1, 2, 3, 4}, {1, 1, 2, 2}, {}, true);
    Tensor b = Tensor::from_vector({5, 6, 7, 8}, {1, 1, 2, 2}, {}, true);
    Tensor c = ops::batch_matmul(a, b);
    Tensor loss = ops::sum(c);
    loss.backward();
    EXPECT_NEAR(a.grad()[0], 11.0, 1e-9);
    EXPECT_NEAR(a.grad()[3], 15.0, 1e-9);
    EXPECT_NEAR(b.grad()[0], 4.0, 1e-9);
    EXPECT_NEAR(b.grad()[3], 6.0, 1e-9);
}

TEST(Linear, BiasBackward) {
    Linear layer(2, 3, true);
    Tensor x = Tensor::from_vector({1, 2}, {1, 2}, {}, false);
    Tensor y = layer.forward(x);
    Tensor loss = ops::sum(y);
    loss.backward();
    EXPECT_EQ(layer.bias.grad().size(), 3u);
    EXPECT_NEAR(layer.bias.grad()[0], 1.0, 1e-9);
    EXPECT_NEAR(layer.bias.grad()[1], 1.0, 1e-9);
    EXPECT_NEAR(layer.bias.grad()[2], 1.0, 1e-9);
}

TEST(Model, Forward) {
    GPTConfig cfg;
    cfg.vocab_size = 100;
    cfg.context_length = 8;
    cfg.emb_dim = 12;
    cfg.n_heads = 3;
    cfg.n_layers = 1;
    GPTModel model(cfg);
    Tensor ids = Tensor::from_ints({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor logits = model.forward(ids);
    EXPECT_EQ(logits.shape(), std::vector<int64_t>({2, 3, 100}));
}

TEST(Tokenizer, Gpt2Bpe) {
    GPT2BPETokenizer tokenizer;
    bool ranks_loaded = tokenizer.load_ranks(std::string(LLM_CPP_SOURCE_DIR) + "/data/gpt2_bpe_ranks.tsv");
    ASSERT_TRUE(ranks_loaded) << "load GPT-2 BPE ranks";
    auto ids = tokenizer.encode("Hello, world!");
    EXPECT_FALSE(ids.empty());
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[0], 15496);
    EXPECT_EQ(ids[1], 11);
    EXPECT_EQ(ids[2], 995);
    EXPECT_EQ(ids[3], 0);
    auto direct = tokenizer.encode("Every effort moves you");
    EXPECT_EQ(direct, std::vector<int64_t>({6109, 3626, 6100, 345}));
}

TEST(Training, Smoke) {
    GPTConfig cfg;
    cfg.vocab_size = 256;
    cfg.context_length = 4;
    cfg.emb_dim = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    GPTModel model(cfg);
    std::vector<int64_t> ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    DataLoader loader(ids, 1, cfg.context_length, cfg.context_length, false);
    AdamW optim(model.parameters(), 1e-3, 0.01);
    double loss = Trainer::train_one_epoch(model, loader, optim);
    EXPECT_GT(loss, 0.0);
}
