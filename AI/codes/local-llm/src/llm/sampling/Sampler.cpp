//
// Created by zhangyoulun on 10/8/2026.
//

#include "Sampler.h"

#include <algorithm>
#include <cmath>
#include <sstream>

std::string SamplingConfig::DebugString() const {
    std::ostringstream os;
    if (is_greedy()) {
        os << "greedy(argmax)";
    } else {
        os << "temperature=" << temperature << " top_k=" << top_k << " top_p=" << top_p
           << " repetition_penalty=" << repetition_penalty << " seed=" << seed;
    }
    return os.str();
}

Sampler::Sampler(const SamplingConfig &config) : config_(config), rng_(config.seed) {}

int Sampler::argmax(const float *logits, const int vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

int Sampler::sample(float *logits, const int vocab, const std::vector<int> &prev_tokens) {
    // 1) 重复惩罚：对上下文中已出现的 token 施加惩罚（HF 约定：正 logit 除以 penalty，
    //    负 logit 乘以 penalty，均使其更不易被选中）。
    if (config_.repetition_penalty > 1.0f && !prev_tokens.empty()) {
        const float p = config_.repetition_penalty;
        for (const int t : prev_tokens) {
            if (t < 0 || t >= vocab) continue;
            float &l = logits[t];
            l = (l > 0.0f) ? (l / p) : (l * p);
        }
    }

    // 贪心：温度<=0 直接取 argmax，忽略 top-k/top-p。
    if (config_.is_greedy()) {
        return argmax(logits, vocab);
    }

    // 2) 温度缩放。
    const float inv_t = 1.0f / config_.temperature;
    for (int i = 0; i < vocab; ++i) {
        logits[i] *= inv_t;
    }

    // 3) 构造候选 (logit, id)。为减少排序开销，先用 top-k 截断（若启用），
    //    否则保留全部。
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;

    const int k = (config_.top_k > 0 && config_.top_k < vocab) ? config_.top_k : vocab;
    if (k < vocab) {
        // 部分排序出前 k 大（按 logit 降序）。
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        idx.resize(k);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    }

    // 4) softmax（对截断后的候选，减最大值稳定）。
    const float max_logit = logits[idx[0]];
    std::vector<float> probs(idx.size());
    float sum = 0.0f;
    for (size_t i = 0; i < idx.size(); ++i) {
        const float e = std::exp(logits[idx[i]] - max_logit);
        probs[i] = e;
        sum += e;
    }
    for (float &pr : probs) pr /= sum;

    // 5) top-p（nucleus）：按降序累加概率，保留累计到 top_p 的最小集合。
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cut = probs.size();
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (cum >= config_.top_p) {
                cut = i + 1;
                break;
            }
        }
        idx.resize(cut);
        probs.resize(cut);
        // 重新归一化。
        float s = 0.0f;
        for (const float pr : probs) s += pr;
        for (float &pr : probs) pr /= s;
    }

    // 6) 按概率采样。
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    const int pick = dist(rng_);
    return idx[pick];
}
