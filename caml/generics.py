import random
import string

from typeguard import typechecked


def generate_random_string(N: int) -> str:
    """
    Function to generate a random string of ascii lowercase letters and digits of length N.

    Utilized to generate a random table name for the Ibis Tables.

    Parameters
    ----------
    N:
        The length of random string to generate.

    Returns
    -------
        str: The random string of length N.
    """
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=N))


def cls_typechecked(cls):
    """
    Class decorator to typecheck all methods of a class.

    Parameters
    ----------
    cls:
        The class to decorate.

    Returns
    -------
        cls: The decorated class.
    """
    for name, func in cls.__dict__.items():
        if callable(func):
            setattr(cls, name, typechecked(func))

    return cls
