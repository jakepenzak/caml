"""Various Utilities for CaML.

This module provides various utilities for CaML.
"""

from importlib.util import find_spec


def is_module_available(module_name: str) -> bool:
    """Check if a module is available.

    Parameters
    ----------
    module_name : str
        The name of the module to check.

    Returns
    -------
    bool
        True if the module is available, False otherwise.
    """
    return find_spec(module_name) is not None
