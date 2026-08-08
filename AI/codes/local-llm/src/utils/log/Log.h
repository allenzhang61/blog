//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_LOG_H
#define LOCAL_LLM_LOG_H
#include <string>


class Log {
public:
    static void debug(const std::string &message);
    static void error(const std::string &message);
};


#endif //LOCAL_LLM_LOG_H
