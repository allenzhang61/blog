#pragma once

#include "llm/tensor.hpp"

namespace llm {

// 语言模型训练用的数据加载器。
// 从一整段 token 序列中切出 input/target，小批量送给模型训练。
class DataLoader {
public:
    // 完整 token 序列。
    std::vector<int64_t> tokens;

    // 每个 batch 包含多少条样本。
    int64_t batch_size;

    // 每条样本的上下文长度。
    int64_t context_length;

    // 滑动窗口步长。
    int64_t stride;

    // 是否打乱样本起点。
    bool shuffle;

    // 输出 Tensor 所在设备。
    Device device{};

    // 当前读取到 starts 中的哪个位置。
    size_t cursor{0};

    // 每个训练样本在 tokens 中的起始下标。
    std::vector<size_t> starts;

    // ids 是完整 token 序列，batch/context/stride 控制切片方式。
    DataLoader(std::vector<int64_t> ids, int64_t batch, int64_t context,
               int64_t stride_, bool shuffle_, Device device_ = {});

    // 重置读取位置；如果 shuffle=true，也会重新打乱样本顺序。
    void reset();

    // 取下一个 batch。
    // input 是当前位置 token，target 是右移一位后的下一个 token。
    bool next(Tensor& input, Tensor& target);
};

} // namespace llm
