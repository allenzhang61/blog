#include "llm/train/Trainer.hpp"

#include "llm/ops.hpp"

namespace llm {

double Trainer::train_one_epoch(GPTModel& model, DataLoader& loader, AdamW& optim) {
    Tensor x, y;
    Tensor last_loss;
    int steps = 0;
    while (loader.next(x, y)) {
        optim.zero_grad();
        Tensor logits = model.forward(x);
        Tensor loss = ops::cross_entropy(logits, y);
        loss.backward();
        optim.step();
        last_loss = loss;
        ++steps;
    }
    if (steps == 0) {
        throw std::runtime_error("no training batches");
    }
    return last_loss.item();
}

double Trainer::train_steps(GPTModel& model, DataLoader& loader, AdamW& optim, int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error("train_steps expects positive steps");
    }
    Tensor x, y;
    Tensor last_loss;
    for (int64_t step = 0; step < steps; ++step) {
        if (!loader.next(x, y)) {
            loader.reset();
            if (!loader.next(x, y)) {
                throw std::runtime_error("no training batches");
            }
        }
        optim.zero_grad();
        Tensor logits = model.forward(x);
        Tensor loss = ops::cross_entropy(logits, y);
        loss.backward();
        optim.step();
        last_loss = loss;
    }
    return last_loss.item();
}

std::vector<int64_t> Trainer::generate_greedy(GPTModel& model, std::vector<int64_t> ids,
                                              int64_t max_new_tokens, int64_t context) {
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        int64_t start = std::max<int64_t>(0, static_cast<int64_t>(ids.size()) - context);
        std::vector<int64_t> window(ids.begin() + start, ids.end());
        Tensor x = Tensor::from_ints(window, {1, static_cast<int64_t>(window.size())}, model.cfg.device);
        Tensor logits = model.forward(x);
        int64_t V = logits.shape()[2];
        int64_t last = logits.shape()[1] - 1;
        int64_t best = 0;
        double best_score = -1e100;
        for (int64_t v = 0; v < V; ++v) {
            double score = logits.data()[(last * V) + v];
            if (score > best_score) {
                best_score = score;
                best = v;
            }
        }
        ids.push_back(best);
    }
    return ids;
}

} // namespace llm
