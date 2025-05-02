import timeit
from functools import wraps
from typing import Protocol, runtime_checkable

import pandas as pd

from .logging import DEBUG, WARNING

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def experimental(obj):
    """
    Decorator to mark functions or classes as experimental.

    This decorator will show a warning when the decorated object is first used,
    indicating that it is experimental and may change in future versions.

    Parameters
    ----------
    obj
        The class or function to mark as experimental

    Returns
    -------
    Union[type, callable]
        The decorated class or function
    """
    warning_msg = f"{obj.__name__} is experimental and may change in future versions."

    # Mark as experimental and initialize warning state
    obj._experimental = True
    obj._experimental_warning_shown = False

    @wraps(obj)
    def wrapper(*args, **kwargs):
        if not obj._experimental_warning_shown:
            WARNING(warning_msg)
            obj._experimental_warning_shown = True
        return obj(*args, **kwargs)

    return wrapper


def timer(operation_name=None):
    """
    Decorator to measure the execution time of a function or method, logged at DEBUG level.

    Parameters
    ----------
    operation_name
        The name of the operation to be timed. If None, the name of the function or method will be used.

    Returns
    -------
    callable
        The decorated function or method
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start = timeit.default_timer()
            result = func(*args, **kwargs)
            end = timeit.default_timer()
            DEBUG(f"{name} completed in {end - start:.2f} seconds")
            return result

        return wrapper

    return decorator if operation_name else decorator(operation_name)


def maybe_jit(func=None, **jit_kwargs):
    def maybe_jit_inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _HAS_JAX:
                return jax.jit(func, **jit_kwargs)(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    if func is None:
        return maybe_jit_inner

    return maybe_jit_inner(func)


@runtime_checkable
class PandasDataFrameConvertible(Protocol):
    """Protocol for DataFrame-like objects that can be converted to pandas."""

    def toPandas(self) -> pd.DataFrame: ...

    def to_pandas(self) -> pd.DataFrame: ...


DataFrameLike = PandasDataFrameConvertible | pd.DataFrame
