//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_MEMORYUSAGEPROVIDER_H
#define LOCAL_LLM_MEMORYUSAGEPROVIDER_H

#include <cstddef>

// 显存用量的通用抽象：任何模型的“一次推理请求作用域”（如 QwenSession）实现此接口，
// 上报本请求持有的两类 device 内存字节数，从而让 MemoryReporter 与具体模型解耦。
//
// 为什么只抽象这两类（而不含权重）：
//   - 权重由通用的 CudaWeightPool 持有，本身与模型无关，MemoryReporter 直接向 pool
//     查询即可，无需经过本接口。
//   - 跨 token 状态与临时激活是 per-request、且各模型形态不同（Qwen 的 KV cache /
//     recurrent state、其它模型可能是别的结构），故由各模型自报字节数。
//
// 后续引入非 Qwen 模型时，只需让其 session 类型实现本接口，MemoryReporter 无需改动。
class MemoryUsageProvider {
public:
    virtual ~MemoryUsageProvider() = default;

    // 跨 token 状态字节数：随序列长度增长的部分（如 KV cache、recurrent / conv state）。
    // 这是 decode 唯一随长度增长的访存量，量化 / f16 KV 的主战场（见 doc/2.md）。
    virtual size_t kv_state_bytes() const = 0;

    // 前向临时激活字节数：grow-only 复用的中间结果峰值（per-request）。
    virtual size_t scratch_bytes() const = 0;
};

#endif // LOCAL_LLM_MEMORYUSAGEPROVIDER_H
