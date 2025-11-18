"""Extensions for CaML including plotting utilities and synthetic data generation.

Import utilities and extensions that work across all data types.
Usage: from caml.extensions import SyntheticDataGenerator, plots
"""

from . import plots
from .synthetic_data import SyntheticDataGenerator

__all__ = [
    "SyntheticDataGenerator",
    "plots",
]
