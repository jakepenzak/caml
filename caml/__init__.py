"""Copyright (c) 2026 Jacob Pieniazek. All rights reserved."""

import warnings

import matplotlib.pyplot as plt
from sklearn.utils.validation import DataConversionWarning

from caml._version import __version__
from caml.automl import AutoCATE

plt.style.use("ggplot")

__all__ = ["__version__", "AutoCATE"]
