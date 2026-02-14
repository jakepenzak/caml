import sys

from . import standard_ml
from .base_estimator import (
    AutoCateEstimator,
    BaseAutoCateEstimatorMixin,
    EstimatorCapabilities,
    InferenceProvider,
)
from .wrappers import dml, dr, meta, orf

# Create aliases so `from caml.estimators.dml import X` works
sys.modules["caml.estimators.dml"] = dml
sys.modules["caml.estimators.dr"] = dr
sys.modules["caml.estimators.meta"] = meta
sys.modules["caml.estimators.orf"] = orf
sys.modules["caml.estimators.standard_ml"] = standard_ml

__all__ = [
    "BaseAutoCateEstimatorMixin",
    "AutoCateEstimator",
    "EstimatorCapabilities",
    "InferenceProvider",
    "dml",
    "dr",
    "meta",
    "orf",
    "standard_ml",
]
