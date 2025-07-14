from typing import Any, Callable, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class _toPandasConvertible(Protocol):
    """Protocol for objects that can be converted to pandas DataFrames."""

    toPandas: Callable[..., Any]

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        """Protocol for DataFrame-like objects that are pandas convertible via `toPandas()`."""
        return callable(getattr(subclass, "toPandas", None))


@runtime_checkable
class _to_pandasConvertible(Protocol):
    """Protocol for DataFrame-like objects that are pandas convertible via `to_pandas()`."""

    to_pandas: Callable[..., Any]

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        """Method to check if a class is a subclass of specs of PandasConvertibleDataFrame."""
        return callable(getattr(subclass, "to_pandas", None))


PandasConvertibleDataFrame = pd.DataFrame | _to_pandasConvertible | _toPandasConvertible


class FittedAttr:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        """Custom getter for attributes that require fitting."""
        if instance is None:
            return self
        if not getattr(instance, "_fitted", False):
            raise RuntimeError("Model has not been fitted yet. Please run fit() first.")
        return getattr(instance, self.name)
