import importlib


def require_module(module_name, install_hint):
    """按需导入依赖；缺失时给出对应安装提示。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"缺少依赖 {module_name}。请先执行：{install_hint}") from exc
