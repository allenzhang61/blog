#pragma once

#include <cstddef>
#include <cstdint>

namespace llm_inference {

void launch_silu_mul(const float * gate, const float * up, float * out, int n, void * stream);
void launch_float_to_bf16(const float * input, uint16_t * output, int n, void * stream);
void launch_rms_norm_to_bf16(const float * input, const uint16_t * weight, uint16_t * output, int n, float eps, bool one_plus, void * stream);

} // namespace llm_inference
