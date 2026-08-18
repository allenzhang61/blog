//
// Created by zhangyoulun on 16/8/2026.
//

#include "format/MF.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

template<typename T>
T MF::metadata(const std::string &key) const {
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

// Metadata variant 覆盖的全部类型：显式实例化，供其他 TU 链接。
template float MF::metadata<float>(const std::string &) const;
template int64_t MF::metadata<int64_t>(const std::string &) const;
template bool MF::metadata<bool>(const std::string &) const;
template std::string MF::metadata<std::string>(const std::string &) const;
template std::vector<int64_t> MF::metadata<std::vector<int64_t>>(const std::string &) const;
template std::vector<std::string> MF::metadata<std::vector<std::string>>(const std::string &) const;

void MF::validate() const {
    const std::vector<std::string> names = tensor_view_names();
    if (names.empty()) {
        throw std::runtime_error("模型文件没有任何 tensor");
    }

    std::set<std::string> seen;
    for (const std::string &name : names) {
        if (name.empty()) {
            throw std::runtime_error("模型文件存在空 tensor name");
        }
        if (!seen.insert(name).second) {
            throw std::runtime_error("模型文件存在重复 tensor: " + name);
        }

        const Tensor &tensor = get_tensor_view(name);
        if (tensor.name != name) {
            throw std::runtime_error("tensor 索引名与视图名不一致: index=" + name +
                                     " view=" + tensor.name);
        }
        if (tensor.shape.empty()) {
            throw std::runtime_error("tensor shape 为空: " + name);
        }
        for (const int64_t dim : tensor.shape) {
            if (dim <= 0) {
                throw std::runtime_error("tensor shape 存在非法维度: " + name +
                                         " shape=" + shape_to_string(tensor.shape));
            }
        }
        if (!is_supported_dtype(tensor.dtype)) {
            throw std::runtime_error("tensor dtype 不受支持: " + name +
                                     " dtype=" + dtype_name(tensor.dtype) +
                                     "（仅支持 F32/F16/BF16/Q4_K/Q5_0/Q6_K/Q8_0）");
        }
        if (tensor.disk_data == nullptr) {
            throw std::runtime_error("tensor data 为空: " + name);
        }
        if (tensor.nbytes == 0) {
            throw std::runtime_error("tensor nbytes 为 0: " + name);
        }
    }
}

void MF::validate_tensor_shape(const std::string &name,
                               const std::vector<int64_t> &expected_shape) const {
    const Tensor &tensor = get_tensor_view(name);
    if (tensor.shape != expected_shape) {
        throw std::runtime_error("tensor shape 不匹配: " + name +
                                 " expected=" + shape_to_string(expected_shape) +
                                 " actual=" + shape_to_string(tensor.shape));
    }
}

std::string MF::shape_to_string(const std::vector<int64_t> &shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}
