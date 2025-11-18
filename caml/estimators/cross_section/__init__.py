"""Cross-sectional data econometric estimators.

Import estimators for cross-sectional data analysis.
Usage: `from caml.cross_section import AutoCATE, InteractiveLinearRegression`
"""

from .cate import AutoCATE, AutoCateEstimator
from .ols import InteractiveLinearRegression

__all__ = [
    "AutoCATE",
    "AutoCateEstimator",
    "InteractiveLinearRegression",
]
