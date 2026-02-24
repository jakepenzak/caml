"""Copyright (c) 2026 Jacob Pieniazek. All rights reserved."""

import matplotlib.pyplot as plt

from caml._version import __version__
from caml.automl import AutoCATE
from caml.logging import configure_logging

plt.style.use("ggplot")

__all__ = ["__version__", "AutoCATE", "configure_logging"]
