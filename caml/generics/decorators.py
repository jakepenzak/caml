"""Decorator utilities for CaML.

This module provides decorators for various functionalities in CaML.
"""

import timeit
from functools import wraps
from typing import Callable

from caml.generics.logging import DEBUG, INFO, WARNING
from caml.generics.utils import is_module_available

_HAS_JAX = is_module_available("jax")

if _HAS_JAX:
    import jax

    jax.config.update("jax_enable_x64", True)
else:
    pass


def experimental(obj: Callable) -> Callable:
    """
    Decorator to mark functions or classes as experimental.

    This decorator will show a warning when the decorated object is first used,
    indicating that it is experimental and may change in future versions.

    Parameters
    ----------
    obj : Callable
        The class or function to mark as experimental

    Returns
    -------
    Callable
        The decorated class or function
    """
    warning_msg = f"{obj.__name__} is experimental and may change in future versions."

    # Mark as experimental and initialize warning state
    obj._experimental = True
    obj._experimental_warning_shown = False

    if isinstance(obj, type):
        # For classes, wrap the __init__ method to show warning after initialization
        original_init = obj.__init__

        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            # Call the original __init__ first
            result = original_init(self, *args, **kwargs)

            # Show warning after __init__ completes (only once per class)
            if not obj._experimental_warning_shown:
                WARNING(warning_msg)
                obj._experimental_warning_shown = True

            return result

        obj.__init__ = wrapped_init
        return obj
    else:
        # For functions, keep the original behavior
        @wraps(obj)
        def wrapper(*args, **kwargs):
            if not obj._experimental_warning_shown:
                WARNING(warning_msg)
                obj._experimental_warning_shown = True
            return obj(*args, **kwargs)

        return wrapper

    def wrapper(*args, **kwargs):
        if not obj._experimental_warning_shown:
            WARNING(warning_msg)
            obj._experimental_warning_shown = True
        return obj(*args, **kwargs)

    return wrapper


def narrate(
    preamble: str | None = None, epilogue: str | None = ":white_check_mark: Completed."
) -> Callable:
    """
    Decorator to log the execution of a function or method.

    This decorator will log a pre-execution (preamble) message and a post-execution (epilogue) message.

    Parameters
    ----------
    preamble : str
        The message to log before the function or method execution.
    epilogue : str
        The message to log after the function or method execution.

    Returns
    -------
    Callable
        The decorated class or function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if preamble is None:
                pass
            else:
                INFO(preamble)
            result = func(*args, **kwargs)
            if epilogue is None:
                pass
            else:
                INFO(epilogue)
            return result

        return wrapper

    return decorator


def timer(operation_name: str | None = None) -> Callable:
    """
    Decorator to measure the execution time of a function or method, logged at DEBUG level.

    Parameters
    ----------
    operation_name : str | None
        The name of the operation to be timed. If None, the name of the function or method will be used.

    Returns
    -------
    Callable
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


def maybe_jit(func: Callable | None = None, **jit_kwargs) -> Callable:
    """Decorator to JIT compile a function using JAX, if available.

    Parameters
    ----------
    func : Callable | None
        The function to be JIT compiled.
    jit_kwargs : dict
        Keyword arguments to be passed to jax.jit.

    Returns
    -------
    Callable
        The decorated function or method
    """

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
