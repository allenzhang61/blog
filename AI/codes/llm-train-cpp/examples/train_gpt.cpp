#include "llm/llm.hpp"

#include <iostream>

int main() {
    using namespace llm;

    GPT2BPETokenizer tokenizer;
    tokenizer.load_ranks(std::string(LLM_CPP_SOURCE_DIR) + "/data/gpt2_bpe_ranks.tsv");
    tokenizer.load_samples(std::string(LLM_CPP_SOURCE_DIR) + "/data/gpt2_bpe_samples.tsv");
    std::vector<int64_t> ids = tokenizer.encode("Every effort moves you forward. Every step teaches.");

    GPTConfig cfg;
    cfg.vocab_size = 50257;
    cfg.context_length = 4;
    cfg.emb_dim = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;

    GPTModel model(cfg);
    DataLoader loader(ids, 1, cfg.context_length, cfg.context_length, false);
    AdamW optim(model.parameters(), 1e-3, 0.01);
    constexpr int64_t kDefaultTrainSteps = 1000;
    double loss = Trainer::train_steps(model, loader, optim, kDefaultTrainSteps);

    std::cout << "train_gpt loss after " << kDefaultTrainSteps << " steps: " << loss << "\n";
    return 0;
}
