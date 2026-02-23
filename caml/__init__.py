"""Copyright (c) 2026 Jacob Pieniazek. All rights reserved."""

import warnings

import matplotlib.pyplot as plt
from sklearn.utils.validation import DataConversionWarning

from caml._version import __version__
from caml.automl import AutoCATE

# Filtering some benign warnings
warnings.filterwarnings(
    "ignore", category=DataConversionWarning, message="A column-vector y was passed"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBM",
)
warnings.filterwarnings("ignore", module="lightgbm")
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
)
plt.style.use("ggplot")

__all__ = ["__version__", "AutoCATE"]
