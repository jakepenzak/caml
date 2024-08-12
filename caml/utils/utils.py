import random
import string


def generate_random_string(N: int) -> str:
    """
    Function to generate a random string of ascii lowercase letters and digits of length N.

    Parameters
    ----------
    N:
        The length of random string to generate.

    Returns
    ----------
        str: The random string of length N.
    """
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=N))
