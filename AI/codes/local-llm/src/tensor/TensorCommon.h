//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_TENSORCOMMON_H
#define LOCAL_LLM_TENSORCOMMON_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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
    I32 = 24,
    BF16 = 30,
    UNKNOWN = -1,
};

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
        case DType::I32: return "I32";
        case DType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

inline bool is_supported_dtype(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::F16:
        case DType::BF16:
        case DType::Q4_K:
        case DType::Q5_0:
        case DType::Q6_K:
        case DType::Q8_0:
            return true;
        default:
            return false;
    }
}

inline size_t tensor_dtype_byte_size(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::I32:
            return 4;
        case DType::F16:
        case DType::BF16:
            return 2;
        default:
            throw std::runtime_error(std::string("dense tensor 不支持 dtype: ") + dtype_name(dt));
    }
}

template <typename T>
inline const char *tensor_cpp_type_name() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float>) {
        return "float";
    } else if constexpr (std::is_same_v<U, int> ||
                         std::is_same_v<U, int32_t>) {
        return "int32";
    } else if constexpr (std::is_same_v<U, uint16_t>) {
        return "uint16";
    } else {
        return "unsupported";
    }
}

template <typename T>
inline bool tensor_dtype_matches_cpp_type(DType dt) {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float>) {
        return dt == DType::F32;
    } else if constexpr (std::is_same_v<U, int> ||
                         std::is_same_v<U, int32_t>) {
        return dt == DType::I32;
    } else if constexpr (std::is_same_v<U, uint16_t>) {
        return dt == DType::F16 || dt == DType::BF16;
    } else {
        return false;
    }
}

template <typename T>
inline void validate_tensor_cpp_type(DType dt, const std::string &name) {
    if (!tensor_dtype_matches_cpp_type<T>(dt)) {
        throw std::runtime_error("tensor data 类型不匹配: " + name +
                                 " dtype=" + dtype_name(dt) +
                                 " requested=" + tensor_cpp_type_name<T>());
    }
}

class TensorShape {
public:
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::UNKNOWN;
    size_t nbytes = 0;

    size_t ndim() const {
        return shape.size();
    }

    int64_t dim(size_t axis) const {
        if (axis >= shape.size()) {
            throw std::runtime_error("tensor shape 维度越界: " + name);
        }
        return shape[axis];
    }

    int64_t numel() const {
        if (shape.empty()) { return 0; }
        int64_t n = 1;
        for (int64_t d : shape) { n *= d; }
        return n;
    }

    int64_t rows() const {
        if (shape.empty()) { return 0; }
        int64_t r = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) { r *= shape[i]; }
        return r;
    }

    int64_t cols() const {
        return shape.empty() ? 0 : shape.back();
    }

    size_t byte_size() const {
        return static_cast<size_t>(numel()) * tensor_dtype_byte_size(dtype);
    }
};

#endif // LOCAL_LLM_TENSORCOMMON_H
