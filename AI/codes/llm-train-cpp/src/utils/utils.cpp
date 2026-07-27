#include "llm/utils.hpp"

#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llm {

int64_t product(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    // 从初始值 1 开始，用乘法把 shape 里的每个维度依次累乘，得到元素总数（如 {2,3,4} -> 24）。
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

int64_t canonical_dim(int64_t dim, int64_t rank) {
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        throw std::runtime_error("dim out of range");
    }
    return dim;
}

} // namespace llm
