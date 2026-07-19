#include "llm/ops.hpp"

namespace llm {
namespace ops {

Tensor softmax(const Tensor& a, int64_t dim) {
    ensure_cpu(a);
    int64_t rank = static_cast<int64_t>(a.shape().size());
    dim = canonical_dim(dim, rank);
    if (dim != rank - 1) throw std::runtime_error("softmax currently supports last dim only");
    int64_t width = a.shape().back();
    int64_t rows = a.numel() / width;
    Tensor out(a.shape(), DType::Float32, a.device(), a.requires_grad());
    for (int64_t r = 0; r < rows; ++r) {
        double mx = -1e100;
        for (int64_t c = 0; c < width; ++c) mx = std::max(mx, a.data()[r * width + c]);
        double denom = 0.0;
        for (int64_t c = 0; c < width; ++c) denom += std::exp(a.data()[r * width + c] - mx);
        for (int64_t c = 0; c < width; ++c) out.data()[r * width + c] = std::exp(a.data()[r * width + c] - mx) / denom;
    }
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, rows, width]() mutable {
            for (int64_t r = 0; r < rows; ++r) {
                double dot = 0.0;
                for (int64_t c = 0; c < width; ++c) dot += out.grad()[r * width + c] * out.data()[r * width + c];
                for (int64_t c = 0; c < width; ++c) {
                    a.grad()[r * width + c] += out.data()[r * width + c] * (out.grad()[r * width + c] - dot);
                }
            }
        };
    }
    return out;
}

Tensor log_softmax(const Tensor& a, int64_t dim) {
    Tensor s = softmax(a, dim);
    Tensor out(a.shape(), DType::Float32, a.device(), a.requires_grad());
    for (int64_t i = 0; i < out.numel(); ++i) out.data()[i] = std::log(std::max(s.data()[i], 1e-12));
    return out;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    ensure_cpu(logits); ensure_cpu(targets);
    if (logits.shape().size() != 3) throw std::runtime_error("cross_entropy expects logits [B,T,V]");
    int64_t B = logits.shape()[0], T = logits.shape()[1], V = logits.shape()[2];
    if (targets.numel() != B * T) throw std::runtime_error("cross_entropy target shape mismatch");
    Tensor out({}, DType::Float32, logits.device(), logits.requires_grad());
    double loss = 0.0;
    std::vector<double> probs(logits.numel(), 0.0);
    for (int64_t row = 0; row < B * T; ++row) {
        double mx = -1e100;
        for (int64_t v = 0; v < V; ++v) mx = std::max(mx, logits.data()[row * V + v]);
        double denom = 0.0;
        for (int64_t v = 0; v < V; ++v) {
            probs[row * V + v] = std::exp(logits.data()[row * V + v] - mx);
            denom += probs[row * V + v];
        }
        int64_t target = static_cast<int64_t>(targets.data()[row]);
        for (int64_t v = 0; v < V; ++v) probs[row * V + v] /= denom;
        loss -= std::log(std::max(probs[row * V + target], 1e-12));
    }
    out.data()[0] = loss / static_cast<double>(B * T);
    if (logits.requires_grad()) {
        out.node->parents = {logits};
        out.node->backward_fn = [logits, targets, out, probs, B, T, V]() mutable {
            for (int64_t row = 0; row < B * T; ++row) {
                int64_t target = static_cast<int64_t>(targets.data()[row]);
                for (int64_t v = 0; v < V; ++v) {
                    double g = probs[row * V + v];
                    if (v == target) g -= 1.0;
                    logits.grad()[row * V + v] += out.grad()[0] * g / static_cast<double>(B * T);
                }
            }
        };
    }
    return out;
}

} // namespace ops
} // namespace llm
