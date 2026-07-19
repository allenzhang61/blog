#include "llm/llm.hpp"

#include <iostream>

int main() {
    using namespace llm;

    GPTConfig cfg;
    cfg.vocab_size = 256;
    cfg.context_length = 4;
    cfg.emb_dim = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;

    GPTModel model(cfg);
    std::vector<int64_t> seed = {65, 66, 67};
    auto ids = Trainer::generate_greedy(model, seed, 3, cfg.context_length);

    std::cout << "generated ids:";
    for (auto id : ids) std::cout << " " << id;
    std::cout << "\n";
    return 0;
}
