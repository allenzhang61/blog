//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_ARGS_H
#define LOCAL_LLM_ARGS_H
#include <string>
#include "backend/common.h"

class Args {
public:
    std::string model_dir;
    Device device = Device::CPU;

    Args(int argc, char **argv);
    void DebugDump();
};


#endif //LOCAL_LLM_ARGS_H
