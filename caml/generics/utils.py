from importlib.util import find_spec


def is_module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


logo = r"""
  ____      __  __ _
 / ___|__ _|  \/  | |
| |   / _` | |\/| | |
| |__| (_| | |  | | |___
 \____\__,_|_|  |_|_____|
"""
