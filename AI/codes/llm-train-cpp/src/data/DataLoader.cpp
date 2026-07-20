#include "llm/data/DataLoader.hpp"

namespace llm {

DataLoader::DataLoader(std::vector<int64_t> ids, int64_t batch, int64_t context,
                       int64_t stride_, bool shuffle_, Device device_)
    : tokens(std::move(ids)), batch_size(batch), context_length(context),
      stride(stride_), shuffle(shuffle_), device(device_) {
    for (size_t i = 0; i + static_cast<size_t>(context_length) < tokens.size(); i += static_cast<size_t>(stride)) {
        starts.push_back(i);
    }
    if (shuffle) {
        std::shuffle(starts.begin(), starts.end(), std::mt19937(123));
    }
}

void DataLoader::reset() {
    cursor = 0;
    if (shuffle) {
        std::shuffle(starts.begin(), starts.end(), std::mt19937(123));
    }
}

bool DataLoader::next(Tensor& input, Tensor& target) {
    if (cursor >= starts.size()) {
        return false;
    }
    int64_t actual = std::min<int64_t>(batch_size, static_cast<int64_t>(starts.size() - cursor));
    std::vector<int64_t> x(actual * context_length), y(actual * context_length);
    for (int64_t b = 0; b < actual; ++b) {
        size_t start = starts[cursor++];
        for (int64_t t = 0; t < context_length; ++t) {
            x[b * context_length + t] = tokens[start + t];
            y[b * context_length + t] = tokens[start + t + 1];
        }
    }
    input = Tensor::from_ints(x, {actual, context_length}, device);
    target = Tensor::from_ints(y, {actual, context_length}, device);
    return true;
}

} // namespace llm
