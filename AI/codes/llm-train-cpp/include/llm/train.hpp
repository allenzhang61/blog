#pragma once

#include "llm/data.hpp"
#include "llm/model.hpp"

namespace llm {

class AdamW {
public:
    explicit AdamW(std::vector<Tensor*> params, double lr = 1e-3, double weight_decay = 0.0,
                   double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
    void zero_grad();
    void step();

private:
    std::vector<Tensor*> params_;
    std::vector<std::vector<double>> m_;
    std::vector<std::vector<double>> v_;
    double lr_;
    double weight_decay_;
    double beta1_;
    double beta2_;
    double eps_;
    int64_t step_{0};
};

class Trainer {
public:
    static double train_one_epoch(GPTModel& model, DataLoader& loader, AdamW& optim);
    static double train_steps(GPTModel& model, DataLoader& loader, AdamW& optim, int64_t steps);
    static std::vector<int64_t> generate_greedy(GPTModel& model, std::vector<int64_t> ids,
                                                int64_t max_new_tokens, int64_t context);
};

} // namespace llm
