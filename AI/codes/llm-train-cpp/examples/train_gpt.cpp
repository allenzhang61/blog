#include "llm/llm.hpp"

#include <fstream>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// 从纯文本文件按行读取 token id（每行一个整数）。
// 该文件由 Python 版用 tiktoken 对 the-verdict.txt 做 BPE 后导出，
// 目的是让 C++ 版与 Python 版使用完全相同的 token 序列，绕开手写 BPE 的差异。
static std::vector<int64_t> load_token_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open token id file: " + path);
    }
    std::vector<int64_t> ids;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ids.push_back(std::stoll(line));
    }
    return ids;
}

int main(int argc, char** argv) {
    try {
        using namespace llm;

        // 用法：train_gpt [backend] [profile] [max_steps] [profile_steps]
        //   backend：cpu / metal / cuda（留空则按环境变量或默认设备选择）。
        //   profile：normal（默认，12 层完整配置）或 small（3 层小配置，快速冒烟/对齐）。
        //   max_steps：可选，限制训练 step 数；0 表示跑完整配置。
        //   profile_steps：可选，非 0 时输出每步耗时拆分，用于调试 backend 性能。
        std::string backend_arg = argc > 1 ? argv[1] : "";
        std::string profile_arg = argc > 2 ? argv[2] : "normal";
        int max_steps = argc > 3 ? std::stoi(argv[3]) : 0;
        bool profile_steps = argc > 4 ? std::stoi(argv[4]) != 0 : false;
        Device device = select_device_from_arg_or_env(backend_arg);
        BackendRegistry::get(device);

        const bool small = (profile_arg == "small");

        // 与 Python 基准对齐：直接使用 tiktoken 导出的 token 序列。
        // small 档用压缩词表版本（token id 重映射到 0..1359），把 vocab 从 50257 缩到 1360，
        // 让 CPU 上最重的输出层（[B*T, vocab] 矩阵）计算量降到约 1/37，单步快几十倍。
        std::vector<int64_t> ids = load_token_ids(
            std::string(LLM_CPP_SOURCE_DIR) +
            (small ? "/data/the_verdict_train_ids_small.txt" : "/data/the_verdict_train_ids.txt"));

        // 两档配置均与 Python 版 main.py 对齐：
        //   normal：GPT_CONFIG_124M（12 层完整配置，vocab=50257）。
        //   small ：小配置（3 层，压缩词表 vocab=1360），跑得快、内存小，用于快速冒烟和逐 step 对齐。
        GPTConfig cfg;
        cfg.vocab_size = small ? 1360 : 50257;
        cfg.context_length = small ? 64 : 256;
        cfg.emb_dim = small ? 128 : 768;
        cfg.n_heads = small ? 4 : 12;
        cfg.n_layers = small ? 3 : 12;
        cfg.device = device;

        GPTModel model(cfg);

        // batch=2, stride=context_length（窗口不重叠），不 shuffle（与 Python 基准脚本对齐）。
        constexpr int64_t kBatchSize = 2;
        DataLoader loader(ids, kBatchSize, cfg.context_length, cfg.context_length, false, device);

        // 优化器对齐 Python：lr=4e-4, weight_decay=0.1（betas/eps 用默认 0.9/0.999/1e-8）。
        AdamW optim(model.parameters(), 4e-4, 0.1);

        const int kNumEpochs = small ? 30 : 10;
        int global_step = -1;
        double last_loss = 0.0;
        bool stop = false;
        for (int epoch = 0; epoch < kNumEpochs; ++epoch) {
            loader.reset();
            Tensor x, y;
            while (loader.next(x, y)) {
                // drop_last：跳过不足一个完整 batch 的尾批，与 Python drop_last=True 对齐。
                if (x.shape()[0] != kBatchSize) {
                    break;
                }
                auto step_start = std::chrono::steady_clock::now();
                optim.zero_grad();
                auto zero_done = std::chrono::steady_clock::now();
                Tensor logits = model.forward(x);
                auto forward_done = std::chrono::steady_clock::now();
                Tensor loss = ops::cross_entropy(logits, y);
                auto loss_done = std::chrono::steady_clock::now();
                loss.backward();
                auto backward_done = std::chrono::steady_clock::now();
                optim.step();
                auto optim_done = std::chrono::steady_clock::now();
                ++global_step;
                last_loss = loss.item();
                auto item_done = std::chrono::steady_clock::now();
                std::cout << "[step " << global_step << "] epoch=" << epoch
                          << " train_loss=" << last_loss << std::endl;
                if (profile_steps) {
                    auto ms = [](auto a, auto b) {
                        return std::chrono::duration<double, std::milli>(b - a).count();
                    };
                    std::cout << "[profile step " << global_step << "]"
                              << " zero_grad_ms=" << ms(step_start, zero_done)
                              << " forward_ms=" << ms(zero_done, forward_done)
                              << " loss_ms=" << ms(forward_done, loss_done)
                              << " backward_ms=" << ms(loss_done, backward_done)
                              << " optim_ms=" << ms(backward_done, optim_done)
                              << " item_ms=" << ms(optim_done, item_done)
                              << " total_ms=" << ms(step_start, item_done)
                              << std::endl;
                }
                if (max_steps > 0 && global_step + 1 >= max_steps) {
                    stop = true;
                    break;
                }
            }
            if (stop) {
                break;
            }
        }

        std::cout << "train_gpt " << device.str() << " final train_loss: " << last_loss << "\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        return 1;
    }
}
