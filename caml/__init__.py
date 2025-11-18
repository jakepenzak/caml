"""Copyright (c) 2024 Jacob Pieniazek. All rights reserved."""

import sys

import matplotlib.pyplot as plt

from caml import extensions
from caml._version import __version__

plt.style.use("ggplot")

from caml.estimators import cross_section, panel, time_series

# Make estimator submodules directly accessible as caml.cross_section, caml.panel, etc.
sys.modules[__name__ + ".cross_section"] = cross_section
sys.modules[__name__ + ".panel"] = panel
sys.modules[__name__ + ".time_series"] = time_series

__all__ = ["__version__", "cross_section", "panel", "time_series", "extensions"]
