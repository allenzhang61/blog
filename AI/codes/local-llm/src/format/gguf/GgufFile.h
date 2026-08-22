//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_GGUFFILE_H
#define LOCAL_LLM_GGUFFILE_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "format/MF.h"

class GGUFTokenizer;

// GGUF（llama.cpp 的模型容器格式）只读解析器，与具体模型结构无关。
// 负责：mmap 打开文件 -> 解析 header / 元数据 KV / 张量表 -> 按名定位张量数据。
// 参考格式规范：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//
// 本类只解析“容器层”（元数据 + 张量位置 + 量化类型），不做反量化；反量化交由
// 上层（CPU 参考实现 / CUDA kernel）按 ggml 量化类型处理。

// ggml 张量数据类型（子集，取自 ggml.h 的 ggml_type 枚举值，值必须与之一致）。
enum class GgmlType : int32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
};

// GGUF 元数据值类型（gguf_metadata_value_type）。
enum class GgufValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// 一个元数据 KV 值。标量存于对应字段；字符串存 str；数组存 arr（元素同 elem_type）。
// 为保持解析器轻量，数组元素统一按其标量语义解释：数值数组用 arr_i64/arr_f64，
// 字符串数组用 arr_str（tokenizer 词表 / merges 等即走这里）。
struct GgufValue {
    GgufValueType type = GgufValueType::UINT32;

    // 标量：数值统一提升到 i64/f64/bool 存放，字符串存 str。
    int64_t i64 = 0;
    double f64 = 0.0;
    bool boolean = false;
    std::string str;

    // 数组：elem_type 为元素类型；数值数组填 arr_i64 或 arr_f64，字符串数组填 arr_str。
    GgufValueType elem_type = GgufValueType::UINT32;
    std::vector<int64_t> arr_i64;
    std::vector<double> arr_f64;
    std::vector<std::string> arr_str;
};

// 顺序读取的游标（相对 mmap 数据起始的字节偏移），解析 GGUF 各段时递进。
class Cursor {
public:
    Cursor(const uint8_t *base, size_t size);

    size_t pos() const { return pos_; }

    // 从 cursor 读取基础类型（little-endian），越界抛异常并前移游标。
    uint8_t read_u8();
    uint32_t read_u32();
    uint64_t read_u64();
    double read_f64_bits(GgufValueType type); // 供 f32/f64 复用

    // GGUF 字符串：u64 长度 + 原始字节。
    std::string read_string();
    // 读取一个元数据值（含数组递归）。
    GgufValue read_value(GgufValueType type);

private:
    const uint8_t *base_ = nullptr;
    size_t size_ = 0;
    size_t pos_ = 0;
};

class GgufFile : public MF {
public:
    // mmap 打开并解析整个 GGUF 文件（header + 元数据 + 张量表）。解析失败抛异常。
    explicit GgufFile(const std::string &path);
    ~GgufFile();

    GgufFile(const GgufFile &) = delete;
    GgufFile &operator=(const GgufFile &) = delete;

    // === 元数据访问 ===
    // 是否存在某个元数据 key。
    bool contain_metadata(const std::string &key) const override;

    // 是否存在某张量。
    bool contain_tensor_view(const std::string &name) const override;

    // === MF tensor 接口 ===
    const StorageTensor &get_tensor_view(const std::string &name) const override;
    std::vector<std::string> tensor_view_names() const override;
    std::vector<int> tokenizer_encode(const std::string &text) const override;
    std::string tokenizer_decode(const std::vector<int> &ids) const override;

    // 打印基础信息（版本、张量数、元数据数、对齐、部分关键 KV）。
    void debug_dump() const override;

private:
    // mmap 状态。
    std::filesystem::path path_;
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t *data_ = nullptr;

    uint32_t version_ = 0;
    uint32_t alignment_ = 32;

    std::map<std::string, GgufValue> metadata_map_;
    std::vector<StorageTensor> tensors_;
    std::unique_ptr<GGUFTokenizer> tokenizer_;

    // GGUF 量化类型 -> 统一 DType。
    static DType gguf_type_to_dtype(GgmlType t);
    Metadata metadata_value(const std::string &key) const override;

    void close();

    // 按 ggml 量化类型计算给定元素数的字节数（block 对齐）。
    static size_t type_nbytes(GgmlType type, int64_t num_elements);
};

#endif // LOCAL_LLM_GGUFFILE_H
