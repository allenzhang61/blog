# 第三方依赖

本项目采用「单头文件入仓」的方式管理第三方库，尽量保持自包含、零构建期外部依赖。

## nlohmann/json

用于解析模型的 `config.json`（如 `QwenConfig`）。

- 引入方式：单头文件（header-only），直接放入仓库。
- 版本：v3.11.3
- 文件位置：`src/thirdparty/nlohmann/json.hpp`
- 许可证：MIT
- 上游仓库：https://github.com/nlohmann/json

### 下载命令

```bash
mkdir -p src/thirdparty/nlohmann
curl -fsSL -o src/thirdparty/nlohmann/json.hpp \
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
```

### 使用方式

`src` 已通过 CMake 加入头文件搜索路径：

```cmake
target_include_directories(local_llm PRIVATE src)
```

因此在源码中以 `src` 为根引用即可：

```cpp
#include "thirdparty/nlohmann/json.hpp"

const nlohmann::json json = nlohmann::json::parse(text);
```

### 升级方式

替换 `src/thirdparty/nlohmann/json.hpp` 为新版本的 `json.hpp` 即可（改上面下载命令里的版本号），无需改动 CMake。
