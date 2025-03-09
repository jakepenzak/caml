import random
import string
from functools import wraps

from typeguard import typechecked

from .logging import WARNING


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


def cls_typechecked(cls):
    """
    Class decorator to typecheck all methods of a class.

    Parameters
    ----------
    cls
        The class to decorate.

    Returns
    -------
    cls
        The decorated class.
    """
    for name, func in cls.__dict__.items():
        if callable(func):
            setattr(cls, name, typechecked(func))

    return cls


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
