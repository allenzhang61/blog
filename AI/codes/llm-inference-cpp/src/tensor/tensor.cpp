#include "tensor.h"

#include "../kernels/cpu/cpu_ops.h"

namespace llm_inference {

DType dtype_from_string(const std::string & s) {
    if (s == "F32") return DType::F32;
    if (s == "BF16") return DType::BF16;
    if (s == "F16") return DType::F16;
    throw std::runtime_error("暂不支持 dtype：" + s);
}

const char * dtype_to_string(DType dt) {
    switch (dt) {
        case DType::F32: return "F32";
        case DType::BF16: return "BF16";
        case DType::F16: return "F16";
    }
    return "?";
}

Tensor Tensor::zeros(std::vector<int64_t> shape) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = DType::F32;
    t.device = Device::CPU;
    t.owned_ = std::make_shared<std::vector<float>>(static_cast<size_t>(t.numel()), 0.0f);
    return t;
}

Tensor Tensor::from_vector(std::vector<int64_t> shape, std::vector<float> data) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = DType::F32;
    t.device = Device::CPU;
    t.owned_ = std::make_shared<std::vector<float>>(std::move(data));
    return t;
}

Tensor Tensor::from_weight(const TensorRef & ref) {
    if (ref.info == nullptr || ref.data == nullptr) {
        throw std::runtime_error("from_weight 收到空 TensorRef");
    }
    Tensor t;
    t.shape = ref.info->shape;
    t.dtype = dtype_from_string(ref.info->dtype);
    t.device = Device::CPU;
    t.ext_ptr_ = ref.data;
    return t;
}

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

float * Tensor::data() {
    if (!owned_ || dtype != DType::F32) {
        throw std::runtime_error("Tensor::data() 仅支持 owned 的 F32 张量");
    }
    return owned_->data();
}

const float * Tensor::data() const {
    if (!owned_ || dtype != DType::F32) {
        throw std::runtime_error("Tensor::data() 仅支持 owned 的 F32 张量");
    }
    return owned_->data();
}

std::vector<float> & Tensor::vec() {
    if (!owned_) throw std::runtime_error("Tensor::vec() 仅支持 owned 张量");
    return *owned_;
}

const std::vector<float> & Tensor::vec() const {
    if (!owned_) throw std::runtime_error("Tensor::vec() 仅支持 owned 张量");
    return *owned_;
}

const void * Tensor::raw() const {
    if (owned_) return owned_->data();
    return ext_ptr_;
}

float Tensor::at(size_t index) const {
    if (owned_) {
        return (*owned_)[index];
    }
    switch (dtype) {
        case DType::BF16:
            return cpu::bf16_to_float(reinterpret_cast<const uint16_t *>(ext_ptr_)[index]);
        case DType::F16:
            return cpu::f16_to_float(reinterpret_cast<const uint16_t *>(ext_ptr_)[index]);
        case DType::F32:
            return reinterpret_cast<const float *>(ext_ptr_)[index];
    }
    throw std::runtime_error("Tensor::at 未知 dtype");
}

} // namespace llm_inference
