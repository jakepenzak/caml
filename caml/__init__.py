"""Copyright (c) 2024 Jacob Pieniazek. All rights reserved."""

import warnings

import matplotlib.pyplot as plt
from sklearn.utils.validation import DataConversionWarning

from caml._version import __version__

# Filtering some benign warnings
warnings.filterwarnings(
    "ignore", category=DataConversionWarning, message="A column-vector y was passed"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBM",
)

plt.style.use("ggplot")

__all__ = ["__version__"]
