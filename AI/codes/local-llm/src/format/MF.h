//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_MF_H
#define LOCAL_LLM_MF_H

#include <string>
#include <variant>
#include <vector>

#include "tensor/CPUTensor.h"
#include "tensor/DiskTensor.h"

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
    virtual const DiskTensor &get_tensor_view(const std::string &name) const = 0;
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

    // 按 key 读取标量/数组元信息。float 与 int64_t 之间允许隐式互转。
    // 仅对 Metadata variant 覆盖的类型做了显式实例化（见 .cpp）。
    template<typename T>
    T metadata(const std::string &key) const;

protected:
    virtual Metadata metadata_value(const std::string &key) const = 0;

private:
    static std::string shape_to_string(const std::vector<int64_t> &shape);
};

#endif // LOCAL_LLM_MF_H
