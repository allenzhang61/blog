//
// Created by zhangyoulun.
//

#ifndef LOCAL_LLM_TENSORCONTAINER_H
#define LOCAL_LLM_TENSORCONTAINER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 模型无关的只读张量容器抽象。
//
// 不同权重文件格式（GGUF、safetensors）在“只读、mmap、按名取张量”这一容器层
// 能力上高度重合，本接口把这部分共性抽象出来，让上层能以统一方式查询/列举张量。
//
// 注意：结构化元数据（如 GGUF 的超参 / tokenizer KV）是 GGUF 独有的，不纳入本接口，
// 由 GgufFile 自身提供；safetensors 的超参在单独的 config.json 中，与本接口无关。

// 统一的张量数据类型枚举，兼容 GGUF 量化类型与 safetensors 浮点类型。
// 数值特意与 ggml_type 保持一致，便于 GgmlType 与本枚举互转。
enum class DType : int32_t {
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
    UNKNOWN = -1,
};

// DType -> 可读名（用于日志/错误信息）。
inline const char *dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        case DType::Q5_0: return "Q5_0";
        case DType::Q5_1: return "Q5_1";
        case DType::Q8_0: return "Q8_0";
        case DType::Q8_1: return "Q8_1";
        case DType::Q2_K: return "Q2_K";
        case DType::Q3_K: return "Q3_K";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::Q8_K: return "Q8_K";
        case DType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

// 单个张量的统一只读视图，不拥有内存（data 指向容器内部的 mmap 区域）。
struct TensorView {
    // 张量名。
    std::string name;
    // 逻辑形状，统一约定为行主序 [out, in]（对二维权重而言）。
    // GGUF 内部 dims 以最内连续维在前，映射到此处时需反转以对齐该约定；
    // safetensors 本就是行主序，直接透传。
    std::vector<int64_t> shape;
    // 数据类型。
    DType dtype = DType::UNKNOWN;
    // 指向 mmap 内张量数据的起始地址。
    const uint8_t *data = nullptr;
    // 张量数据字节数。
    size_t nbytes = 0;
};

// 只读张量容器抽象基类。GgufFile / SafeTensorsFile 各自实现。
class TensorContainer {
public:
    virtual ~TensorContainer() = default;

    // 是否存在某张量。
    virtual bool contains(const std::string &name) const = 0;
    // 按名返回张量视图；不存在时抛异常。
    virtual const TensorView &get(const std::string &name) const = 0;
    // 全部张量名。
    virtual std::vector<std::string> names() const = 0;
    // 打印容器元信息（不含原始权重数值）。
    virtual void DebugDump() const = 0;
};

#endif // LOCAL_LLM_TENSORCONTAINER_H
