#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm {

enum class DeviceType { CPU, CUDA, Metal };
enum class DType { Float32, Int64 };

std::string to_string(DeviceType type);
std::string to_string(DType dtype);

struct Device {
    DeviceType type{DeviceType::CPU};
    int index{0};

    static Device parse(const std::string& text);
    std::string str() const;
};

int64_t product(const std::vector<int64_t>& shape);
int64_t canonical_dim(int64_t dim, int64_t rank);

void check(bool cond, const std::string& message);
void check_close(double a, double b, double tol, const std::string& message);

} // namespace llm
