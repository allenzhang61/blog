#pragma once

#include "cli.h"
#include "common.h"
#include "config.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace llm_inference {

// 加载 vocab.json，并生成 token id 到 token piece 的反向表。
std::unordered_map<int, std::string> load_vocab_reverse(const fs::path & model_dir, double & elapsed);

// 将 token ids 通过 vocab 反查并拼接为文本。
std::string detokenize(const std::vector<int> & ids, const std::unordered_map<int, std::string> & vocab);

// 根据 Args 选择 --input-ids、prompt 或默认 prompt ids。
std::vector<int> resolve_input_ids(const Args & args);

} // namespace llm_inference
