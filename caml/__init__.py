"""Copyright (c) 2024 Jacob Pieniazek. All rights reserved."""

import sys

import matplotlib.pyplot as plt

from caml._version import __version__

plt.style.use("ggplot")

from caml.extensions import plots, synthetic_data

__all__ = ["__version__", "plots", "synthetic_data"]
