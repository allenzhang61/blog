#pragma once

namespace llm_inference {

// 运行时 CUDA tensor 的轻量引用，不拥有 device 内存。
// 和 WeightData 不同，它描述的是 forward 过程中流动的激活或 token buffer。
struct Tensor {
    // CUDA device 指针。
    void * data = nullptr;
    // 元素数量，不包含 dtype 字节数。
    int elements = 0;
    // scratch buffer slot；不属于固定 scratch buffer 时为 -1。
    int slot = -1;
};

} // namespace llm_inference
