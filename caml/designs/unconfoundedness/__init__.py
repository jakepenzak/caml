"""Cross-sectional data econometric estimators.

Import estimators for cross-sectional data analysis.
Usage: `from caml.cross_section import AutoCATE, InteractiveLinearRegression`
"""

from . import InteractiveLinearRegression
from .auto_cate import AutoCATE, AutoCateEstimator

__all__ = [
    "AutoCATE",
    "AutoCateEstimator",
    "InteractiveLinearRegression",
]
