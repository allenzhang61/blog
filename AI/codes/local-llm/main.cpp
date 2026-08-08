#include <iostream>

#include "utils/cli/Args.h"
#include "llm/qwen/QwenConfig.h"
#include "llm/qwen/QwenTokenizer.h"
#include "llm/qwen/QwenWeights.h"

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
    std::cout << tokenizer.Decode(tokenizer.Encode("hi")) << std::endl;

    std::vector<int> inputs = tokenizer.Encode("法国的首都是");


    return 0;
    // TIP See CLion help at <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>. Also, you can try interactive lessons for CLion by selecting 'Help | Learn IDE Features' from the main menu.
}
