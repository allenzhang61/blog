//
// Created by zhangyoulun on 9/8/2026.
//

#include "Q4K.h"

// 【注意】本文件当前未被任何代码调用（死代码），保留作为 Q4_K 反量化的 CPU 参考实现，
// 用于对拍验证 CUDA kernel 的正确性。详见 Q4K.h 顶部说明。

#include <cstdint>
#include <cstring>
#include <stdexcept>

float q4k_half_to_float(uint16_t h) {
    // 标准 IEEE-754 half -> float 转换。
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; // ±0
        } else {
            // 非规格化：规格化到 float。
            int e = -1;
            uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); // Inf / NaN
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void q4k_get_scale_min(int j, const uint8_t *scales, uint8_t &sc, uint8_t &m) {
    // 与 llama.cpp get_scale_min_k4 一致：前 4 个 sub-block 的 6-bit 值直接取低 6 位，
    // 后 4 个 sub-block 的高 2 位借用前段字节的高 2 位拼接。
    if (j < 4) {
        sc = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

void q4k_dequantize_block(const BlockQ4K *block, float *out) {
    const float d = q4k_half_to_float(block->d);
    const float min = q4k_half_to_float(block->dmin);
    const uint8_t *qs = block->qs;

    int out_idx = 0;
    // 8 个 sub-block，每个 32 元素；每 2 个 sub-block 共享 32 字节 qs（低/高 4-bit）。
    for (int j = 0; j < kQ4K_SubBlocks; j += 2) {
        uint8_t sc_lo, m_lo, sc_hi, m_hi;
        q4k_get_scale_min(j, block->scales, sc_lo, m_lo);
        q4k_get_scale_min(j + 1, block->scales, sc_hi, m_hi);

        const float d1 = d * sc_lo, min1 = min * m_lo; // 低 4-bit 的 sub-block
        const float d2 = d * sc_hi, min2 = min * m_hi; // 高 4-bit 的 sub-block

        for (int i = 0; i < 32; ++i) {
            out[out_idx + i] = d1 * (qs[i] & 0x0F) - min1;
        }
        for (int i = 0; i < 32; ++i) {
            out[out_idx + 32 + i] = d2 * (qs[i] >> 4) - min2;
        }
        qs += 32;
        out_idx += 64;
    }
}

void q4k_dequantize_row(const uint8_t *src, float *out, int64_t num_elements) {
    if (num_elements % kQ4K_BlockSize != 0) {
        throw std::runtime_error("Q4_K 反量化：元素数不是 256 的整数倍");
    }
    const int64_t nblocks = num_elements / kQ4K_BlockSize;
    const auto *blocks = reinterpret_cast<const BlockQ4K *>(src);
    for (int64_t b = 0; b < nblocks; ++b) {
        q4k_dequantize_block(&blocks[b], out + b * kQ4K_BlockSize);
    }
}
