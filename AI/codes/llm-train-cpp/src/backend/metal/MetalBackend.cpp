#include "llm/backend/Backend.hpp"
#include "llm/metal_ops.hpp"

namespace llm {

bool metal_backend_available() {
    return false;
}

std::string metal_backend_status() {
    return "Metal backend is unavailable: project was not compiled with LLM_CPP_ENABLE_METAL=ON on an Apple platform";
}

} // namespace llm

namespace llm::metal {

bool available() {
    return false;
}

std::string status() {
    return metal_backend_status();
}

Tensor add(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor sub(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor mul(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor div(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor mul_scalar(const Tensor&, double) {
    throw std::runtime_error(status());
}

Tensor pow(const Tensor&, double) {
    throw std::runtime_error(status());
}

Tensor sum(const Tensor&) {
    throw std::runtime_error(status());
}

Tensor mean(const Tensor&) {
    throw std::runtime_error(status());
}

Tensor max(const Tensor&) {
    throw std::runtime_error(status());
}

Tensor reshape(const Tensor&, const std::vector<int64_t>&) {
    throw std::runtime_error(status());
}

Tensor transpose(const Tensor&, int64_t, int64_t) {
    throw std::runtime_error(status());
}

Tensor matmul(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor batch_matmul(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor softmax(const Tensor&, int64_t) {
    throw std::runtime_error(status());
}

Tensor log_softmax(const Tensor&, int64_t) {
    throw std::runtime_error(status());
}

Tensor cross_entropy(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor embedding(const Tensor&, const Tensor&) {
    throw std::runtime_error(status());
}

Tensor layernorm(const Tensor&, const Tensor&, const Tensor&, double) {
    throw std::runtime_error(status());
}

Tensor gelu(const Tensor&) {
    throw std::runtime_error(status());
}

} // namespace llm::metal
