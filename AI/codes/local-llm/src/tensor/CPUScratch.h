//
// Created by zhangyoulun on 26/8/2026.
//

#ifndef LOCAL_LLM_CPUSCRATCH_H
#define LOCAL_LLM_CPUSCRATCH_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Host 端临时缓冲区（grow-only 复用），用于给 CPUTensor 提供由 session 持有的内存。
class CPUScratch {
public:
    CPUScratch() = default;

    CPUScratch(const CPUScratch &) = delete;
    CPUScratch &operator=(const CPUScratch &) = delete;

    template <typename T>
    T *ensure(const std::string &key, size_t count) {
        std::vector<std::max_align_t> &buf = buffers_[key];
        const size_t required_bytes = count * sizeof(T);
        const size_t required_slots = (required_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
        if (buf.size() < required_slots) {
            buf.resize(required_slots);
        }
        return reinterpret_cast<T *>(buf.data());
    }

    void reset() { buffers_.clear(); }

private:
    std::unordered_map<std::string, std::vector<std::max_align_t>> buffers_;
};

namespace cpu_scratch_key {
inline constexpr const char *kInputIds = "input_ids";
inline constexpr const char *kLmHeadLogits = "lm_head_logits";
inline constexpr const char *kMoeExpertIds = "moe_expert_ids";
inline constexpr const char *kMoeWeights = "moe_weights";
} // namespace cpu_scratch_key

#endif // LOCAL_LLM_CPUSCRATCH_H
