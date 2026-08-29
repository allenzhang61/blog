#ifndef LOCAL_LLM_DEEPSEEK_TRACE_H
#define LOCAL_LLM_DEEPSEEK_TRACE_H

#include <string>

class DeepseekSession;
class GPUTensor;

namespace deepseek_trace {
bool enabled();
bool pos_match(int pos);
const char *tag();

void tensor(DeepseekSession &session, const GPUTensor &g_tensor,
            const char *stage, int pos, int layer, int row = -1);
void topk(DeepseekSession &session, const GPUTensor &g_idx_i32, const GPUTensor &g_w_f32,
          const char *stage, int pos, int layer, int row = -1);
} // namespace deepseek_trace

#endif // LOCAL_LLM_DEEPSEEK_TRACE_H
