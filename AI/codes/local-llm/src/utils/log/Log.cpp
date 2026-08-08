//
// Created by zhangyoulun on 8/8/2026.
//

#include "Log.h"

#include <iostream>
#include <string>

void Log::debug(const std::string &message) {
    std::cout << message << std::endl;
}

void Log::error(const std::string &message) {
    std::cerr << message << std::endl;
}
