#pragma once

#include "cli.h"
#include "config.h"
#include "safetensors.h"
#include "tensor_ops.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llm_inference {

// 推理流程各阶段耗时和 token 统计。
struct Timing {
    // 加载 config.json 的耗时。
    double load_config_s = 0.0;
    // mmap 加载 safetensors 权重的耗时。
    double load_weights_s = 0.0;
    // 加载 vocab.json 的耗时。
    double load_vocab_s = 0.0;
    // 校验 Qwen tensor 完整性的耗时。
    double validate_s = 0.0;
    // prefill 阶段耗时。
    double prefill_s = 0.0;
    // decode 阶段累计耗时。
    double decode_total_s = 0.0;
    // logits / argmax 阶段累计耗时。
    double logits_s = 0.0;
    // 预热运行总耗时。
    double warmup_s = 0.0;
    // 正式推理墙钟耗时。
    double infer_wall_s = 0.0;
    // 输入 token 数。
    int input_tokens = 0;
    // 已生成 token 数。
    int generated_tokens = 0;
    // 已生成 token ids，用于 profile 输出。
    std::vector<int> generated_ids;
};

// 转义字符串，生成可嵌入 JSON 的内容。
std::string json_escape(const std::string & value);

// 将 tensor shape 转为 "[d0, d1, ...]" 字符串。
std::string shape_to_string(const std::vector<int64_t> & shape);

// 汇总配置、权重和耗时信息，生成 profile JSON。
std::string profile_json(const ModelConfig & config, const ModelWeights & weights, const Timing & timing, const Args & args);

// 打印当前权重中所有 tensor 的名称、dtype、shape 和文件信息。
void dump_tensors(const ModelWeights & weights);

} // namespace llm_inference
