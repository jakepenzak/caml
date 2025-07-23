from importlib.util import find_spec


def is_module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None
