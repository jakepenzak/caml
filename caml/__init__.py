"""Copyright (c) 2024 Jacob Pieniazek. All rights reserved."""

import matplotlib.pyplot as plt

from caml._version import __version__
from caml.core.cate import AutoCATE
from caml.core.modeling.model_bank import AutoCateEstimator
from caml.core.ols import FastOLS

plt.style.use("ggplot")
