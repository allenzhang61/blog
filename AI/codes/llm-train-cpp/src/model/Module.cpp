#include "llm/model/Module.hpp"

namespace llm {

std::vector<Tensor*> Module::parameters() {
    return {};
}

void Module::zero_grad() {
    for (auto* p : parameters()) {
        p->zero_grad();
    }
}

} // namespace llm
