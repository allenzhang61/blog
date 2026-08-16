//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_MF_H
#define LOCAL_LLM_MF_H

#include <cstddef>
#include <cstdint>
#include <variant>
#include <string>
#include <type_traits>
#include <vector>

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

// 单个张量的统一只读视图，不拥有内存（data 指向模型文件 mmap 区域）。
struct MFTensorView {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::UNKNOWN;
    const uint8_t *data = nullptr;
    size_t nbytes = 0;
};

using Metadata = std::variant<int64_t, float, std::string, bool,
                              std::vector<int64_t>, std::vector<std::string>>;

// 模型文件/目录的共同抽象：
//   - tensor 访问：按名称访问权重张量
//   - tokenizer_encode/tokenizer_decode：文本与 token id 的互转
//
// 具体格式可以是 HF 目录（config.json/tokenizer.json/safetensors），也可以是 GGUF
// 单文件（metadata/tensors/tokenizer metadata）。模型层优先依赖这个接口，而不是直接
// 依赖 HFFile/GgufFile。
// model file
class MF {
public:
    virtual ~MF() = default;

    // 是否存在某张量。
    virtual bool contain_tensor_view(const std::string &name) const = 0;
    // 按名返回张量视图；不存在时抛异常。
    virtual const MFTensorView &get_tensor_view(const std::string &name) const = 0;
    // 全部张量名。
    virtual std::vector<std::string> tensor_view_names() const = 0;

    virtual std::vector<int> tokenizer_encode(const std::string &text) const = 0;
    virtual std::string tokenizer_decode(const std::vector<int> &ids) const = 0;

    // 打印模型文件元信息（不含原始权重数值）。
    virtual void debug_dump() const = 0;

    // 校验模型文件容器层信息是否自洽：张量名唯一、shape 合法、dtype 已支持、
    // data/nbytes 不为空。模型结构相关的 shape 校验可继续用 validate_tensor_shape。
    void validate() const;

    // 校验某个 tensor 的逻辑 shape 是否符合上层模型配置。
    void validate_tensor_shape(const std::string &name,
                               const std::vector<int64_t> &expected_shape) const;

    virtual bool contain_metadata(const std::string &key) const = 0;

    template<typename T>
    T metadata(const std::string &key) const {
        const Metadata value = metadata_value(key);
        if constexpr (std::is_same_v<T, float>) {
            if (const auto *v = std::get_if<float>(&value)) {
                return *v;
            }
            if (const auto *v = std::get_if<int64_t>(&value)) {
                return static_cast<float>(*v);
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (const auto *v = std::get_if<int64_t>(&value)) {
                return *v;
            }
            if (const auto *v = std::get_if<float>(&value)) {
                return static_cast<int64_t>(*v);
            }
        } else {
            return std::get<T>(value);
        }
        return std::get<T>(value);
    }

protected:
    virtual Metadata metadata_value(const std::string &key) const = 0;

private:
    static std::string shape_to_string(const std::vector<int64_t> &shape);
};

#endif // LOCAL_LLM_MF_H
