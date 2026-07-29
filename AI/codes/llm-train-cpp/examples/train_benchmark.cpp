#include "llm/llm.hpp"

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    try {
        using namespace llm;

        std::string backend = argc > 1 ? argv[1] : "cpu";
        int64_t steps = argc > 2 ? std::stoll(argv[2]) : 100;
        int64_t batch = argc > 3 ? std::stoll(argv[3]) : 8;
        int64_t context = argc > 4 ? std::stoll(argv[4]) : 16;
        int64_t emb = argc > 5 ? std::stoll(argv[5]) : 64;
        int64_t heads = argc > 6 ? std::stoll(argv[6]) : 4;
        int64_t layers = argc > 7 ? std::stoll(argv[7]) : 1;
        int64_t vocab = argc > 8 ? std::stoll(argv[8]) : 1024;
        if (steps <= 0 || batch <= 0 || context <= 0 || emb <= 0 || heads <= 0 || layers <= 0 || vocab <= 1) {
            throw std::runtime_error(
                "usage: train_benchmark <cpu|metal|cuda> [steps] [batch] [context] [emb] [heads] [layers] [vocab]");
        }
        if (emb % heads != 0) {
            throw std::runtime_error("emb must be divisible by heads");
        }

        Device device = select_device_from_arg_or_env(backend);

        std::vector<int64_t> ids(static_cast<size_t>((steps + 2) * batch * context + 1));
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<int64_t>((i * 13 + 7) % static_cast<size_t>(vocab));
        }

        GPTConfig cfg;
        cfg.vocab_size = vocab;
        cfg.context_length = context;
        cfg.emb_dim = emb;
        cfg.n_heads = heads;
        cfg.n_layers = layers;
        cfg.device = device;

        GPTModel model(cfg);
        DataLoader loader(ids, batch, context, context, false, device);
        AdamW optim(model.parameters(), 1e-3, 0.01);

        auto start = std::chrono::steady_clock::now();
        double loss = Trainer::train_steps(model, loader, optim, steps);
        auto end = std::chrono::steady_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double tokens = static_cast<double>(steps * batch * context);
        std::cout << "train_benchmark backend=" << device.str()
                  << " steps=" << steps
                  << " batch=" << batch
                  << " context=" << context
                  << " emb=" << emb
                  << " heads=" << heads
                  << " layers=" << layers
                  << " vocab=" << vocab
                  << " loss=" << loss
                  << " seconds=" << seconds
                  << " tokens_per_second=" << (tokens / seconds)
                  << "\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        return 1;
    }
}
