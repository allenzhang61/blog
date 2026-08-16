//
// Created by zhangyoulun on 16/8/2026.
//

#include "format/MF.h"

#include <set>
#include <sstream>
#include <stdexcept>

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

        const MFTensorView &tensor = get_tensor_view(name);
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
        if (tensor.data == nullptr) {
            throw std::runtime_error("tensor data 为空: " + name);
        }
        if (tensor.nbytes == 0) {
            throw std::runtime_error("tensor nbytes 为 0: " + name);
        }
    }
}

void MF::validate_tensor_shape(const std::string &name,
                               const std::vector<int64_t> &expected_shape) const {
    const MFTensorView &tensor = get_tensor_view(name);
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
