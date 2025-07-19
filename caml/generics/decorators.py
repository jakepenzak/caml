import timeit
from functools import wraps
from typing import Callable

from caml.generics.utils import is_module_available
from caml.logging import DEBUG, WARNING

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

    @wraps(obj)
    def wrapper(*args, **kwargs):
        if not obj._experimental_warning_shown:
            WARNING(warning_msg)
            obj._experimental_warning_shown = True
        return obj(*args, **kwargs)

    return wrapper


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
