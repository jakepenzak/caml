"""Various Utilities for CaML.

This module provides various utilities for CaML.
"""

from importlib.util import find_spec

import numpy as np


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


def arr_at_least_2d(x):
    x2 = np.atleast_2d(x)
    if (
        x2.shape[0] == 1
    ):  # Assuming 1D row vector (will need to update if support more treatments or outcomes)
        x2 = x2.T
    return x2
