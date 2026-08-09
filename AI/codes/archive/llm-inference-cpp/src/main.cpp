#include "main.h"

#include <exception>
#include <iostream>

using namespace llm_inference;

int main(int argc, char ** argv) {
    try {
        Args args = parse_args(argc, argv);
        const fs::path model_dir(args.model_dir);
        Timing timing;

        log(std::string("开始加载 ") + MODEL_ID + " 原生 C++ 权重 ...");
        auto start = Clock::now();
        const ModelConfig config = load_config(model_dir);
        timing.load_config_s = elapsed_s(start);

        start = Clock::now();
        ModelWeights weights = ModelWeights::load_mmap(model_dir);
        timing.load_weights_s = elapsed_s(start);

        start = Clock::now();
        weights.validate_qwen_tensors(config);
        timing.validate_s = elapsed_s(start);

        double vocab_s = 0.0;
        auto vocab = load_vocab_reverse(model_dir, vocab_s);
        timing.load_vocab_s = vocab_s;

        if (args.dump_tensors) {
            weights.dump_tensors();
        }

        const std::vector<int> input_ids = resolve_input_ids(args);
        timing.input_tokens = static_cast<int>(input_ids.size());

        if (args.warmup_runs > 0) {
            log("开始预热，次数 " + std::to_string(args.warmup_runs) + "，不计入正式推理耗时...");
            start = Clock::now();
            for (int i = 0; i < args.warmup_runs; ++i) {
                Timing warm_timing;
                warm_timing.input_tokens = timing.input_tokens;
                QwenModel model(config, weights);
                QwenGenerator generator(model, config);
                RunState warm_state(config, warm_timing.input_tokens + args.max_new_tokens + 4);
                (void) generator.generate(warm_state, args, input_ids, warm_timing);
            }
            timing.warmup_s = elapsed_s(start);
            log("预热完成，耗时 " + std::to_string(timing.warmup_s) + "s");
        }

        log("开始推理...");
        start = Clock::now();
        QwenModel model(config, weights);
        QwenGenerator generator(model, config);
        RunState state(config, timing.input_tokens + args.max_new_tokens + 4);
        std::vector<int> generated = generator.generate(state, args, input_ids, timing);
        timing.infer_wall_s = elapsed_s(start);
        log("推理完成，耗时 " + std::to_string(timing.infer_wall_s) +
            "s，max_new_tokens=" + std::to_string(args.max_new_tokens));

        if (args.profile_timing) {
            log("PROFILE_TIMING_JSON:");
            log(profile_json(config, weights, timing, args));
        }

        std::cout << detokenize(generated, vocab) << std::endl;
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "推理失败：" << exc.what() << std::endl;
        return 1;
    }
}
