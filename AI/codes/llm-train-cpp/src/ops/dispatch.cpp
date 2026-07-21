#include "llm/ops.hpp"

#include "llm/cpu_ops.hpp"
#include "llm/metal_ops.hpp"

namespace llm {
namespace ops {

// ops 层只负责按设备把算子分发到具体后端（cpu:: / metal::），本身不做计算。

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.device().type == DeviceType::Metal || b.device().type == DeviceType::Metal) {
        if (a.device().type != DeviceType::Metal || b.device().type != DeviceType::Metal) {
            throw std::runtime_error("add expects tensors on the same device");
        }
        return metal::add(a, b);
    }
    return cpu::add(a, b);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.device().type == DeviceType::Metal || b.device().type == DeviceType::Metal) {
        if (a.device().type != DeviceType::Metal || b.device().type != DeviceType::Metal) {
            throw std::runtime_error("sub expects tensors on the same device");
        }
        return metal::sub(a, b);
    }
    return cpu::sub(a, b);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.device().type == DeviceType::Metal || b.device().type == DeviceType::Metal) {
        if (a.device().type != DeviceType::Metal || b.device().type != DeviceType::Metal) {
            throw std::runtime_error("mul expects tensors on the same device");
        }
        return metal::mul(a, b);
    }
    return cpu::mul(a, b);
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.device().type == DeviceType::Metal || b.device().type == DeviceType::Metal) {
        if (a.device().type != DeviceType::Metal || b.device().type != DeviceType::Metal) {
            throw std::runtime_error("div expects tensors on the same device");
        }
        return metal::div(a, b);
    }
    return cpu::div(a, b);
}

Tensor mul_scalar(const Tensor& a, double scalar) {
    if (a.device().type == DeviceType::Metal) {
        return metal::mul_scalar(a, scalar);
    }
    return cpu::mul_scalar(a, scalar);
}

Tensor pow(const Tensor& a, double exponent) {
    if (a.device().type == DeviceType::Metal) {
        return metal::pow(a, exponent);
    }
    return cpu::pow(a, exponent);
}

Tensor sum(const Tensor& a) {
    if (a.device().type == DeviceType::Metal) {
        return metal::sum(a);
    }
    return cpu::sum(a);
}

Tensor mean(const Tensor& a) {
    if (a.device().type == DeviceType::Metal) {
        return metal::mean(a);
    }
    return cpu::mean(a);
}

Tensor max(const Tensor& a) {
    if (a.device().type == DeviceType::Metal) {
        return metal::max(a);
    }
    return cpu::max(a);
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    if (a.device().type == DeviceType::Metal) {
        return metal::reshape(a, new_shape);
    }
    return cpu::reshape(a, new_shape);
}

Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
    if (a.device().type == DeviceType::Metal) {
        return metal::transpose(a, dim0, dim1);
    }
    return cpu::transpose(a, dim0, dim1);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.device().type == DeviceType::Metal || b.device().type == DeviceType::Metal) {
        if (a.device().type != DeviceType::Metal || b.device().type != DeviceType::Metal) {
            throw std::runtime_error("matmul expects tensors on the same device");
        }
        return metal::matmul(a, b);
    }
    return cpu::matmul(a, b);
}

Tensor batch_matmul(const Tensor& a, const Tensor& b) {
    if (a.device().type != b.device().type) {
        throw std::runtime_error("batch_matmul expects tensors on the same device");
    }
    if (a.device().type == DeviceType::Metal) {
        return metal::batch_matmul(a, b);
    }
    return cpu::batch_matmul(a, b);
}

Tensor softmax(const Tensor& a, int64_t dim) {
    if (a.device().type == DeviceType::Metal) {
        return metal::softmax(a, dim);
    }
    return cpu::softmax(a, dim);
}

Tensor log_softmax(const Tensor& a, int64_t dim) {
    if (a.device().type == DeviceType::Metal) {
        return metal::log_softmax(a, dim);
    }
    return cpu::log_softmax(a, dim);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    if (logits.device().type != targets.device().type) {
        throw std::runtime_error("cross_entropy expects logits and targets on the same device");
    }
    if (logits.device().type == DeviceType::Metal) {
        return metal::cross_entropy(logits, targets);
    }
    return cpu::cross_entropy(logits, targets);
}

Tensor embedding(const Tensor& ids, const Tensor& weight) {
    if (ids.device().type != weight.device().type) {
        throw std::runtime_error("embedding expects ids and weight on the same device");
    }
    if (ids.device().type == DeviceType::Metal) {
        return metal::embedding(ids, weight);
    }
    return cpu::embedding(ids, weight);
}

Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    if (x.device().type != scale.device().type || x.device().type != shift.device().type) {
        throw std::runtime_error("layernorm expects tensors on the same device");
    }
    if (x.device().type == DeviceType::Metal) {
        return metal::layernorm(x, scale, shift, eps);
    }
    return cpu::layernorm(x, scale, shift, eps);
}

Tensor gelu(const Tensor& x) {
    if (x.device().type == DeviceType::Metal) {
        return metal::gelu(x);
    }
    return cpu::gelu(x);
}

} // namespace ops
} // namespace llm
