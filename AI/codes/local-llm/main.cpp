#include <iostream>
#include <chrono>

#include "utils/cli/Args.h"
#include "llm/qwen/QwenConfig.h"
#include "llm/qwen/QwenSession.h"
#include "llm/qwen/QwenTokenizer.h"
#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/model/QwenModel.h"
#include "backend/cuda/mem/CudaWeightPool.h"

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.

int main(int argc, char **argv) {
    // TIP Press <shortcut actionId="RenameElement"/> when your caret is at the <b>lang</b> variable name to see how CLion can help you rename it.
    Args args(argc, argv);
    args.DebugDump();

    QwenConfig config(args.model_dir + "/config.json");
    config.DebugDump();

    QwenWeights weights(args.model_dir, config);
    weights.DebugDump();

    QwenTokenizer tokenizer(args.model_dir + "/tokenizer.json");
    tokenizer.DebugDump();

    std::vector<int> inputs = tokenizer.Encode("法国的首都是");

    QwenSession session(config, inputs, args.max_output_tokens);

    // device 权重缓存（惰性上传）+ 模型（串起各 Module，无 per-request 状态）。
    CudaWeightPool pool;
    QwenModel model(config, weights, &pool);

    const int eos = config.data.text.eos_token_id;

    using Clock = std::chrono::steady_clock;
    auto elapsed_s = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };

    // warmup：正式计时前跑一遍完整前向（独立 session），触发权重惰性上传、
    // CUDA context / cuBLAS 初始化与 kernel 首次启动，避免首步冷启动污染稳态计时。
    {
        QwenSession warm(config, inputs, 4);
        int wnext = model.prefill(warm, inputs);
        int wpos = static_cast<int>(inputs.size());
        for (int i = 0; i < 4 && wnext != eos; ++i) {
            warm.h_outputs.push_back(wnext);
            wnext = model.decode(warm, wnext, wpos);
            ++wpos;
        }
    }

    // prefill：喂入整段 prompt，跑完 32 层，返回首个生成 token。
    auto prefill_start = Clock::now();
    int next = model.prefill(session, inputs);
    double prefill_s = elapsed_s(prefill_start);

    // decode：从 prompt 之后的位置开始逐 token 生成。
    int pos = static_cast<int>(inputs.size());
    int decode_tokens = 0;
    auto decode_start = Clock::now();
    for (int step = 0; step < args.max_output_tokens; ++step) {
        if (next == eos) break;
        session.h_outputs.push_back(next);
        next = model.decode(session, next, pos);
        ++pos;
        ++decode_tokens;
    }
    double decode_total_s = elapsed_s(decode_start);

    std::cout << "生成结果：" << tokenizer.Decode(session.h_outputs) << std::endl;

    std::cout << "PROFILE input_tokens=" << inputs.size()
              << " prefill_s=" << prefill_s
              << " decode_tokens=" << decode_tokens
              << " decode_total_s=" << decode_total_s
              << " decode_per_token_ms=" << (decode_tokens > 0 ? decode_total_s / decode_tokens * 1000.0 : 0.0)
              << " decode_tokens_per_s=" << (decode_total_s > 0 ? decode_tokens / decode_total_s : 0.0)
              << std::endl;
    return 0;
    // TIP See CLion help at <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>. Also, you can try interactive lessons for CLion by selecting 'Help | Learn IDE Features' from the main menu.
}
