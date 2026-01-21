from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from caml.data.dataset import CausalDataset
from caml.data.schema import Estimand, OutcomeType, TreatmentType
from caml.inference.schema import InferenceType


@dataclass(frozen=True)
class EstimatorCapabilites:
    """Describes what an estimator supports."""

    treatment_types: set[TreatmentType]
    outcome_types: set[OutcomeType]
    inference_types: set[InferenceType]
    estimands: set[Estimand]
    requires_propensity: bool = False
    supports_inference: bool = False


@runtime_checkable
class CATEEstimator(Protocol):
    """Core protocol for CATE Estimators."""

    capabilities: EstimatorCapabilites

    def fit(self, data: CausalDataset, **kwargs) -> "CATEEstimator":
        """Fit the estimator."""
        ...

    def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict CATE for given features."""
        ...

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters (sklearn compatability)."""
        ...

    def set_params(self, **params) -> dict:
        """Set parameters (sklearn compatability)."""
        ...
