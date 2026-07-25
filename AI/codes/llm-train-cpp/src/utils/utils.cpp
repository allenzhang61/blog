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

void check(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error("check failed: " + message);
    }
}

void check_close(double a, double b, double tol, const std::string& message) {
    if (std::fabs(a - b) > tol) {
        std::ostringstream oss;
        oss << message << " expected " << b << " got " << a;
        throw std::runtime_error(oss.str());
    }
}

} // namespace llm
