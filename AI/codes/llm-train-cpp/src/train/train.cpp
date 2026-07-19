#include "llm/train.hpp"

namespace llm {

AdamW::AdamW(std::vector<Tensor*> params, double lr, double weight_decay, double beta1, double beta2, double eps)
    : params_(std::move(params)),
      lr_(lr),
      weight_decay_(weight_decay),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps) {
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (const auto* p : params_) {
        m_.push_back(std::vector<double>(static_cast<size_t>(p->numel()), 0.0));
        v_.push_back(std::vector<double>(static_cast<size_t>(p->numel()), 0.0));
    }
}

void AdamW::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

void AdamW::step() {
    ++step_;
    double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_));
    double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(step_));
    for (size_t pi = 0; pi < params_.size(); ++pi) {
        auto* p = params_[pi];
        if (p->grad().empty()) continue;
        for (int64_t i = 0; i < p->numel(); ++i) {
            double g = p->grad()[i];
            m_[pi][i] = beta1_ * m_[pi][i] + (1.0 - beta1_) * g;
            v_[pi][i] = beta2_ * v_[pi][i] + (1.0 - beta2_) * g * g;
            double m_hat = m_[pi][i] / bias_correction1;
            double v_hat = v_[pi][i] / bias_correction2;
            p->data()[i] -= lr_ * weight_decay_ * p->data()[i];
            p->data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

double Trainer::train_one_epoch(GPTModel& model, DataLoader& loader, AdamW& optim) {
    Tensor x, y;
    double last_loss = 0.0;
    int steps = 0;
    while (loader.next(x, y)) {
        optim.zero_grad();
        Tensor logits = model.forward(x);
        Tensor loss = ops::cross_entropy(logits, y);
        last_loss = loss.item();
        loss.backward();
        optim.step();
        ++steps;
    }
    if (steps == 0) throw std::runtime_error("no training batches");
    return last_loss;
}

double Trainer::train_steps(GPTModel& model, DataLoader& loader, AdamW& optim, int64_t steps) {
    if (steps <= 0) throw std::runtime_error("train_steps expects positive steps");
    Tensor x, y;
    double last_loss = 0.0;
    for (int64_t step = 0; step < steps; ++step) {
        if (!loader.next(x, y)) {
            loader.reset();
            if (!loader.next(x, y)) throw std::runtime_error("no training batches");
        }
        optim.zero_grad();
        Tensor logits = model.forward(x);
        Tensor loss = ops::cross_entropy(logits, y);
        last_loss = loss.item();
        loss.backward();
        optim.step();
    }
    return last_loss;
}

std::vector<int64_t> Trainer::generate_greedy(GPTModel& model, std::vector<int64_t> ids,
                                              int64_t max_new_tokens, int64_t context) {
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        int64_t start = std::max<int64_t>(0, static_cast<int64_t>(ids.size()) - context);
        std::vector<int64_t> window(ids.begin() + start, ids.end());
        Tensor x = Tensor::from_ints(window, {1, static_cast<int64_t>(window.size())});
        Tensor logits = model.forward(x);
        int64_t V = logits.shape()[2];
        int64_t last = logits.shape()[1] - 1;
        int64_t best = 0;
        double best_score = -1e100;
        for (int64_t v = 0; v < V; ++v) {
            double score = logits.data()[(last * V) + v];
            if (score > best_score) { best_score = score; best = v; }
        }
        ids.push_back(best);
    }
    return ids;
}

} // namespace llm
