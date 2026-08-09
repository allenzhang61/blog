#include <iostream>

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

    // prefill：喂入整段 prompt，跑完 32 层，返回首个生成 token。
    int next = model.prefill(session, inputs);

    // decode：从 prompt 之后的位置开始逐 token 生成。
    int pos = static_cast<int>(inputs.size());
    for (int step = 0; step < args.max_output_tokens; ++step) {
        if (next == eos) break;
        session.h_outputs.push_back(next);
        next = model.decode(session, next, pos);
        ++pos;
    }

    std::cout << "生成结果：" << tokenizer.Decode(session.h_outputs) << std::endl;
    return 0;
    // TIP See CLion help at <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>. Also, you can try interactive lessons for CLion by selecting 'Help | Learn IDE Features' from the main menu.
}
