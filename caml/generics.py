import random
import string
import timeit
from functools import wraps

from .logging import DEBUG, WARNING


def generate_random_string(N: int) -> str:
    """
    Function to generate a random string of ascii lowercase letters and digits of length N.

    Utilized to generate a random table name for the Ibis Tables.

    Parameters
    ----------
    N
        The length of random string to generate.

    Returns
    -------
    str
        The random string of length N.
    """
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=N))


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
