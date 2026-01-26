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


class FittedAttr:
    """Attribute that requires `_fitted` attribute to be True."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        """Custom getter for attributes that require fitting."""
        if instance is None:
            return self
        if not getattr(instance, "_fitted", False):
            raise RuntimeError("Model has not been fitted yet. Please run fit() first.")
        return getattr(instance, self.name)
