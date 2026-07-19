#include "llm/llm.hpp"

#include <iostream>

using namespace llm;

void test_tensor_basics() {
    Tensor x = Tensor::ones({2, 3});
    check(x.numel() == 6, "numel");
    check(x.shape()[0] == 2 && x.shape()[1] == 3, "shape");
    check(to_string(x.dtype()) == "float32", "dtype");
    check(x.device().type == DeviceType::CPU, "device");
}

void test_backend_placeholders() {
    check(BackendRegistry::get(Device::parse("cpu")).name() == "CPUBackend", "cpu backend");
    bool cuda_failed = false;
    try {
        BackendRegistry::get(Device::parse("cuda"));
    } catch (const std::runtime_error& err) {
        cuda_failed = std::string(err.what()).find("CUDA backend is not implemented") != std::string::npos;
    }
    check(cuda_failed, "cuda placeholder");

    bool metal_failed = false;
    try {
        BackendRegistry::get(Device::parse("metal"));
    } catch (const std::runtime_error& err) {
        metal_failed = std::string(err.what()).find("Metal backend is not implemented") != std::string::npos;
    }
    check(metal_failed, "metal placeholder");
}

void test_ops_and_autograd() {
    Tensor a = Tensor::from_vector({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor b = Tensor::from_vector({5, 6, 7, 8}, {2, 2}, {}, true);
    Tensor c = ops::matmul(a, b);
    check(c.shape() == std::vector<int64_t>({2, 2}), "matmul shape");
    check_close(c.data()[0], 19.0, 1e-9, "matmul value 0");
    check_close(c.data()[3], 50.0, 1e-9, "matmul value 3");

    Tensor loss = ops::sum(c);
    loss.backward();
    check(a.grad().size() == a.data().size(), "a grad exists");
    check(b.grad().size() == b.data().size(), "b grad exists");
    check_close(a.grad()[0], 11.0, 1e-9, "matmul backward a00");
    check_close(b.grad()[0], 4.0, 1e-9, "matmul backward b00");
}

void test_neural_ops() {
    Tensor logits = Tensor::from_vector({1, 2, 3, 1, 3, 2}, {1, 2, 3}, {}, true);
    Tensor targets = Tensor::from_ints({2, 1}, {1, 2});
    Tensor loss = ops::cross_entropy(logits, targets);
    check(loss.item() > 0.0, "cross entropy positive");
    loss.backward();
    check(logits.grad().size() == logits.data().size(), "cross entropy grad");

    Tensor x = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    Tensor s = ops::softmax(x, -1);
    check_close(s.data()[0] + s.data()[1], 1.0, 1e-9, "softmax row 0");
    check_close(s.data()[2] + s.data()[3], 1.0, 1e-9, "softmax row 1");
}

void test_batch_matmul_backward() {
    Tensor a = Tensor::from_vector({1, 2, 3, 4}, {1, 1, 2, 2}, {}, true);
    Tensor b = Tensor::from_vector({5, 6, 7, 8}, {1, 1, 2, 2}, {}, true);
    Tensor c = ops::batch_matmul(a, b);
    Tensor loss = ops::sum(c);
    loss.backward();
    check_close(a.grad()[0], 11.0, 1e-9, "batch_matmul backward a00");
    check_close(a.grad()[3], 15.0, 1e-9, "batch_matmul backward a11");
    check_close(b.grad()[0], 4.0, 1e-9, "batch_matmul backward b00");
    check_close(b.grad()[3], 6.0, 1e-9, "batch_matmul backward b11");
}

void test_linear_bias_backward() {
    Linear layer(2, 3, true);
    Tensor x = Tensor::from_vector({1, 2}, {1, 2}, {}, false);
    Tensor y = layer.forward(x);
    Tensor loss = ops::sum(y);
    loss.backward();
    check(layer.bias.grad().size() == 3, "linear bias grad exists");
    check_close(layer.bias.grad()[0], 1.0, 1e-9, "linear bias grad 0");
    check_close(layer.bias.grad()[1], 1.0, 1e-9, "linear bias grad 1");
    check_close(layer.bias.grad()[2], 1.0, 1e-9, "linear bias grad 2");
}

void test_model_forward() {
    GPTConfig cfg;
    cfg.vocab_size = 100;
    cfg.context_length = 8;
    cfg.emb_dim = 12;
    cfg.n_heads = 3;
    cfg.n_layers = 1;
    GPTModel model(cfg);
    Tensor ids = Tensor::from_ints({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor logits = model.forward(ids);
    check(logits.shape() == std::vector<int64_t>({2, 3, 100}), "GPTModel logits shape");
}

void test_tokenizer() {
    GPT2BPETokenizer tokenizer;
    bool ranks_loaded = tokenizer.load_ranks(std::string(LLM_CPP_SOURCE_DIR) + "/data/gpt2_bpe_ranks.tsv");
    check(ranks_loaded, "load GPT-2 BPE ranks");
    bool loaded = tokenizer.load_samples(std::string(LLM_CPP_SOURCE_DIR) + "/data/gpt2_bpe_samples.tsv");
    check(loaded, "load GPT-2 BPE samples");
    auto ids = tokenizer.encode("Hello, world!");
    check(!ids.empty(), "tokenizer ids");
    check(ids.size() == 4, "gpt2 sample length");
    check(ids[0] == 15496 && ids[1] == 11 && ids[2] == 995 && ids[3] == 0, "gpt2 sample ids");
    auto direct = tokenizer.encode("Every effort moves you");
    check(direct == std::vector<int64_t>({6109, 3626, 6100, 345}), "gpt2 direct bpe ids");
}

void test_training_smoke() {
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
    check(loss > 0.0, "training loss");
}

int main() {
    try {
        test_tensor_basics();
        test_backend_placeholders();
        test_ops_and_autograd();
        test_neural_ops();
        test_batch_matmul_backward();
        test_linear_bias_backward();
        test_model_forward();
        test_tokenizer();
        test_training_smoke();
        std::cout << "all llm-train-cpp tests passed\n";
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        return 1;
    }
    return 0;
}
