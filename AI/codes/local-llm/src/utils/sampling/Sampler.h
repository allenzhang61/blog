//
// Created by zhangyoulun on 10/8/2026.
//

#ifndef LOCAL_LLM_SAMPLER_H
#define LOCAL_LLM_SAMPLER_H

#include <cstdint>
#include <random>
#include <string>
#include <vector>

// 采样配置：从命令行解析后一路透传到各模型。
// 约定：
//   - temperature <= 0 视为贪心（argmax），忽略 top_k / top_p，等价旧行为。
//   - top_k <= 0 表示不启用 top-k 截断。
//   - top_p >= 1.0（或 <= 0）表示不启用 top-p 截断。
//   - repetition_penalty <= 1.0 表示不惩罚（1.0 为中性）。
struct SamplingConfig {
    float temperature = 0.0f;        // 0 -> greedy
    int top_k = 0;                   // 0 -> disabled
    float top_p = 1.0f;              // >=1 -> disabled
    float repetition_penalty = 1.0f; // <=1 -> disabled
    uint64_t seed = 42;

    bool is_greedy() const { return temperature <= 0.0f; }
    std::string DebugString() const;
};

// host 端采样器：输入一整行 logits（长度 vocab），可选带上已生成 token 历史用于
// 重复惩罚，返回采样得到的下一个 token id。
//
// 为什么放在 host：logits 每步只有 vocab 个 float（~100k-150k），D2H 拷贝相对整条
// 前向可忽略；host 端实现 top-k/top-p/温度/重复惩罚逻辑清晰、无需新增 CUDA kernel，
// 也天然可在无 CUDA 环境编译测试。
class Sampler {
public:
    explicit Sampler(const SamplingConfig &config);

    // logits：device 计算后拷回 host 的一整行（会被本函数原地修改，调用方勿复用）。
    // vocab ：logits 长度。
    // prev_tokens：本次生成上下文已出现的 token（prompt + 已生成），用于重复惩罚；
    //             传空则不惩罚。
    // 返回：下一个 token id。
    int sample(float *logits, int vocab, const std::vector<int> &prev_tokens);

    // 是否为贪心（argmax）采样。贪心时 decode 可走 GPU argmax + CUDA Graph 闭环。
    bool is_greedy() const { return config_.is_greedy(); }

private:
    static int argmax(const float *logits, int vocab);

    SamplingConfig config_;
    std::mt19937_64 rng_;
};

#endif // LOCAL_LLM_SAMPLER_H
