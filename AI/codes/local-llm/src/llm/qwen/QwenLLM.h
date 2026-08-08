//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_QWENLLM_H
#define LOCAL_LLM_QWENLLM_H
#include <string>
#include <vector>

class QwenLLM {
public:
    std::vector<int> inference(std::vector<int> &inputs);

    void prefill(std::vector<int> &inputs);
    void decode();
};


#endif //LOCAL_LLM_QWENLLM_H
