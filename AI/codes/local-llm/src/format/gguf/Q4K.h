//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_Q4K_H
#define LOCAL_LLM_Q4K_H

// ============================================================================
// 【注意】本模块当前未被任何代码调用（死代码），暂保留作为参考实现。
//
// 实际推理中的 Q4_K 反量化走的是 CUDA kernel（src/backend/cuda/ops/kernel.cu），
// 与此处 CPU 实现相互独立；kernel.cu 只在注释里“参考”了 BlockQ4K 的布局，
// 并未包含本头文件或引用这里的符号。
//
// 保留它的意义在于“对拍验证”（differential testing / 交叉验证）：
//   同一段 Q4_K 权重字节，分别用两套独立实现反量化——
//     A = 本文件的 CPU 参考实现 q4k_dequantize_row（简单、直白、几乎不会写错）
//     B = CUDA kernel 的 GPU 反量化（快，但并行代码容易写错）
//   逐元素比较 A 与 B（允许极小浮点误差）：
//     一致  -> 说明难写的 GPU kernel 正确；
//     不一致 -> 说明 GPU kernel 有 bug。
//   即用简单可信的 CPU 版当“标准答案”，来校验难调试的 GPU 版。
//
// 目前对拍单测尚未落地，因此这套 CPU 参考实现处于闲置状态。
// 若确定不再需要对拍，可连同 CMakeLists.txt 里的引用一并删除。
// ============================================================================

#include <cstddef>
#include <cstdint>

// Q4_K 量化格式定义与 CPU 参考反量化，与 llama.cpp 的 block_q4_K / dequantize_row_q4_K
// 保持一致，用于：
//   1) 单元/对拍验证 CUDA kernel 的正确性；
//   2) 上层加载时按需在 host 侧反量化（若需要）。
//
// Q4_K super-block：256 个元素为一组，包含
//   - d    : super-block 的 scale（f16）
//   - dmin : super-block 的 min（f16）
//   - scales[12] : 8 个 sub-block（每 32 元素）各自的 6-bit scale 与 6-bit min，打包在 12 字节内
//   - qs[128]    : 256 个 4-bit 量化值（每字节 2 个）
// 反量化：对第 j 个 sub-block（32 元素），第 i 个元素
//   y = d * sc_j * q - dmin * m_j
// 其中 sc_j / m_j 为 6-bit scale/min，q 为该元素的 4-bit 值。

constexpr int kQ4K_BlockSize = 256;      // 一个 super-block 的元素数
constexpr int kQ4K_BlockBytes = 144;     // 一个 super-block 的字节数
constexpr int kQ4K_SubBlocks = 8;        // 每 super-block 的 sub-block 数

// 与 llama.cpp 内存布局逐字节一致的 super-block 结构（不含填充）。
#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;          // f16 scale
    uint16_t dmin;       // f16 min
    uint8_t scales[12];  // 8×(6-bit scale, 6-bit min) 打包
    uint8_t qs[128];     // 256×4-bit
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == kQ4K_BlockBytes, "BlockQ4K 布局必须为 144 字节");

// 把 IEEE-754 half（f16）位模式转成 float。
float q4k_half_to_float(uint16_t h);

// 从 12 字节 scales 中解出第 j 个 sub-block 的 6-bit scale 与 min（与 llama.cpp get_scale_min_k4 一致）。
void q4k_get_scale_min(int j, const uint8_t *scales, uint8_t &sc, uint8_t &m);

// 反量化一个 super-block（256 个元素）到 out（float）。
void q4k_dequantize_block(const BlockQ4K *block, float *out);

// 反量化整段 Q4_K 数据：num_elements 必须是 256 的整数倍。
//   src : Q4_K 原始字节（连续的 BlockQ4K）；
//   out : 输出 float 数组，长度 >= num_elements。
void q4k_dequantize_row(const uint8_t *src, float *out, int64_t num_elements);

#endif // LOCAL_LLM_Q4K_H
