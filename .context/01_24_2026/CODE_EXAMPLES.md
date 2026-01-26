# CaML Refactoring - Complete Code Examples

This document provides detailed code examples for every file in the proposed directory structure.

**Note**: Due to the large size, this file contains abbreviated examples. Full implementations should follow these patterns.

---

## Table of Contents

1. [data/](#1-data)
2. [protocols/](#2-protocols)
3. [estimators/](#3-estimators)
4. [nuisance/](#4-nuisance)
5. [scoring/](#5-scoring)
6. [validation/](#6-validation)
7. [automl/](#7-automl)
8. [inference/](#8-inference)
9. [modeling/](#9-modeling)
10. [benchmarking/](#10-benchmarking)

---

## 1. data/

### data/__init__.py

```python
"""Data containers and validation."""
from caml.data.dataset import CausalDataset
from caml.data.data_schema import TreatmentType, OutcomeType, Estimand

__all__ = ["CausalDataset", "TreatmentType", "OutcomeType", "Estimand"]
```

### data/data_schema.py

**Status**: ✅ **IMPLEMENTED**

```python
from enum import Enum


class TreatmentType(Enum):
    """Categories of treatment variables."""

    BINARY = "binary"
    MULTI = "multi"
    CONTINUOUS = "continuous"

    def is_discrete(self) -> bool:
        return self in {TreatmentType.BINARY, TreatmentType.MULTI}


class OutcomeType(Enum):
    """Categories of outcome variables."""

    BINARY = "binary"
    CONTINUOUS = "continuous"

    def is_discrete(self) -> bool:
        return self == OutcomeType.BINARY


class Estimand(Enum):
    """Categories of target estimands."""

    ATE = "ate"  # Average Treatment Effect
    ATT = "att"  # Average Treatment Effect on Treated
    ATC = "atc"  # Average Treatment Effect on Control
    CATE = "cate"  # Conditional ATE
    GATE = "gate"  # Group ATE
```

### data/dataset.py

**Status**: ✅ **IMPLEMENTED**

The actual implementation is in `/home/jadmin/projects/caml/caml/data/dataset.py` and matches the design from REFACTORING_PLAN.md with:
- Core data fields (X, T, Y, W)
- Metadata (treatment_type, outcome_type)
- Feature names tracking
- `validate()` method
- `from_dataframe()` class method
- Automatic validation on initialization via `__post_init__`

### data/_validation.py

**Status**: ✅ **IMPLEMENTED**

Complete validation utilities including:
- `check_shapes_match()` - Verify all arrays have compatible shapes
- `check_1d_targets()` - Ensure treatment and outcome are 1-dimensional
- `check_missing_data()` - Check for missing values and return summary
- `check_treatment_type_matches_data()` - Verify declared treatment type matches data
- `check_outcome_type_matches_data()` - Verify declared outcome type matches data
- Helper functions for robust array handling (pandas/numpy)

---

## 2. protocols/

### protocols/__init__.py

```python
"""Protocol definitions for estimators and inference providers."""
from caml.protocols.estimator import CATEEstimator, EstimatorCapabilities
from caml.protocols.inference import InferenceProvider

__all__ = ["CATEEstimator", "EstimatorCapabilities", "InferenceProvider"]
```

### protocols/estimator.py

**Status**: ✅ **IMPLEMENTED**

Key implementation details:
- `EstimatorCapabilities` dataclass (frozen) with:
  - `treatment_types`: set of supported TreatmentType
  - `outcome_types`: set of supported OutcomeType
  - `inference_types`: set of supported InferenceType
  - `estimands`: set of supported Estimand
  - `requires_propensity`: bool flag
  - `supports_inference`: bool flag
  - `is_compatible(data)`: method to check dataset compatibility

- `CATEEstimator` Protocol with:
  - `capabilities`: EstimatorCapabilities property
  - `fit(data: CausalDataset, **kwargs)`: fit method
  - `effect(X, **kwargs)`: predict CATE method (note: uses "effect" not "predict_cate")
  - `get_params(deep=True)`: sklearn-compatible param getter
  - `set_params(**params)`: sklearn-compatible param setter

**Note**: The actual implementation uses `effect()` instead of `predict_cate()` as the main prediction method.

### protocols/inference.py

**Status**: ✅ **IMPLEMENTED**

Comprehensive `InferenceProvider` Protocol with three methods:
1. `effect_interval(X, alpha, method, **kwargs)` -> tuple[lower, upper]
   - Supports 'auto', 'analytic', 'bootstrap' methods
   - Returns confidence interval bounds

2. `effect_stderr(X, method, **kwargs)` -> np.ndarray
   - Returns standard errors for each observation

3. `effect_inference(X, alpha, method, **kwargs)` -> InferenceResult
   - Returns complete InferenceResult with point estimates, CIs, stderr, metadata

Method parameter allows flexibility:
- 'auto': Use estimator's preferred method
- 'analytic': Use analytic/asymptotic standard errors
- 'bootstrap': Use bootstrap resampling

---

## 3. estimators/

### estimators/__init__.py

```python
"""CATE estimators."""
from caml.estimators.benchmark.interactive_ols import InteractiveLinearRegression

__all__ = ["InteractiveLinearRegression"]
```

### estimators/base.py

```python
"""Base utilities (optional)."""
from sklearn.base import clone

class BaseEstimatorMixin:
    def clone(self):
        return clone(self)
```

### estimators/benchmark/__init__.py

```python
"""Benchmark estimators."""
from caml.estimators.benchmark.interactive_ols import InteractiveLinearRegression

__all__ = ["InteractiveLinearRegression"]
```

### estimators/benchmark/interactive_ols.py

**Status**: ⚠️ **PARTIALLY IMPLEMENTED** - Exists but needs protocol adaptation

Current status:
- File exists at `/home/jadmin/projects/caml/caml/estimators/benchmark/interactive_ols.py` (24,781 bytes)
- Implements existing `BaseCamlEstimator` and `OLSMixin` patterns
- Uses Patsy formula-based design matrix creation
- Supports formula-based model specification with interaction terms
- Methods include: `fit()`, `predict()`, `_estimate_ate()`, `_estimate_gate()`, `_estimate_cate()`

**Refactoring needed**:
1. Add `capabilities` property returning `EstimatorCapabilities`
2. Adapt `fit()` to accept `CausalDataset` (currently expects DataFrame)
3. Add/adapt `effect()` method to match `CATEEstimator` protocol
4. Ensure `get_params()`/`set_params()` are compatible with sklearn interface
5. Consider adding `InferenceProvider` protocol implementation (analytic inference already supported)

**Current signature** (to be adapted):
```python
class InteractiveLinearRegression(BaseCamlEstimator, OLSMixin):
    def __init__(self, Y, T, G=None, X=None, W=None, xformula=None, discrete_treatment=False):
        # Formula-based initialization

    def fit(self, df: pd.DataFrame):
        # Currently expects DataFrame directly

    def predict(self, df, mode='outcome'):
        # mode can be 'outcome', 'cate', 'ate', 'gate'
```

**Target signature** (protocol-compliant):
```python
class InteractiveLinearRegression(BaseCamlEstimator, OLSMixin):
    @property
    def capabilities(self) -> EstimatorCapabilities:
        return EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
            inference_types={InferenceType.ANALYTIC},
            estimands={Estimand.ATE, Estimand.GATE, Estimand.CATE},
            supports_inference=True
        )

    def fit(self, data: CausalDataset, **kwargs):
        # Adapt to use CausalDataset

    def effect(self, X, **kwargs):
        # Map to existing predict(mode='cate')
```

### estimators/wrappers/__init__.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

```python
"""EconML estimator wrappers."""
# TODO: Import wrapper modules once implemented
# from caml.estimators.wrappers import dml, dr, meta, orf

__all__ = ["dml", "dr", "meta", "orf"]
```

### estimators/wrappers/dml.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

**Target implementation** (from REFACTORING_PLAN.md):

```python
"""Wrappers for EconML DML estimators."""

from econml.dml import (
    LinearDML,
    SparseLinearDML,
    CausalForestDML,
    NonParamDML,
    KernelDML
)
from caml.protocols.estimator import EstimatorCapabilities
from caml.data.data_schema import TreatmentType, OutcomeType
from caml.data.dataset import CausalDataset
import numpy as np
import pandas as pd


class WrappedLinearDML:
    """Wrapper for EconML's LinearDML."""

    def __init__(self, **econml_kwargs):
        # Set defaults
        defaults = {
            "model_y": "auto",
            "model_t": "auto",
            "cv": 3,
            "discrete_treatment": False
        }
        defaults.update(econml_kwargs)

        self.estimator = LinearDML(**defaults)
        self._econml_kwargs = defaults

        self._capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.MULTI, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_propensity=False,
            supports_inference=True,
            inference_method="analytic",
            requires_W=True
        )

    @property
    def capabilities(self):
        return self._capabilities

    def fit(self, data: CausalDataset, **kwargs):
        """Fit using CausalDataset."""
        # Update discrete_treatment flag based on data
        if data.treatment_type.is_discrete():
            self.estimator.discrete_treatment = True

        # Prepare data
        X = data.X if isinstance(data.X, (pd.DataFrame, np.ndarray)) else data.X.values
        T = data.T if isinstance(data.T, (pd.Series, np.ndarray)) else data.T.values
        Y = data.Y if isinstance(data.Y, (pd.Series, np.ndarray)) else data.Y.values
        W = data.W if data.W is not None else None

        # Fit
        self.estimator.fit(
            Y=Y,
            T=T,
            X=X if X is not None and (isinstance(X, pd.DataFrame) and not X.empty or (isinstance(X, np.ndarray) and X.size > 0)) else None,
            W=W if W is not None and (isinstance(W, pd.DataFrame) and not W.empty or (isinstance(W, np.ndarray) and W.size > 0)) else None,
            **kwargs
        )

        return self

    def predict_cate(self, X, **kwargs):
        """Predict CATE."""
        return self.estimator.effect(X, **kwargs)

    def predict_interval(self, X, alpha=0.05, **kwargs):
        """Predict confidence interval."""
        lower, upper = self.estimator.effect_interval(X, alpha=alpha, **kwargs)
        return lower, upper

    def predict_stderr(self, X, **kwargs):
        """Predict standard errors."""
        return self.estimator.effect_stderr(X, **kwargs)

    def get_params(self, deep=True):
        return self._econml_kwargs.copy()

    def set_params(self, **params):
        self._econml_kwargs.update(params)
        self.estimator.set_params(**params)
        return self


class WrappedSparseLinearDML:
    """Wrapper for EconML's SparseLinearDML."""
    # Similar structure to WrappedLinearDML
    pass


class WrappedCausalForestDML:
    """Wrapper for EconML's CausalForestDML."""
    # Similar structure
    pass


class WrappedNonParamDML:
    """Wrapper for EconML's NonParamDML."""
    # Similar structure
    pass


class WrappedKernelDML:
    """Wrapper for EconML's KernelDML."""
    # Similar structure
    pass
```

### estimators/wrappers/dr.py

```python
"""Wrappers for EconML DR (doubly-robust) estimators."""

from econml.dr import DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from caml.protocols.estimator import EstimatorCapabilities
from caml.data.data_schema import TreatmentType, OutcomeType
# ... same pattern as dml.py


class WrappedDRLearner:
    """Wrapper for EconML's DRLearner."""

    def __init__(self, **econml_kwargs):
        defaults = {
            "model_propensity": "auto",
            "model_regression": "auto",
            "model_final": "auto",
            "cv": 3
        }
        defaults.update(econml_kwargs)

        self.estimator = DRLearner(**defaults)
        self._econml_kwargs = defaults

        self._capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.MULTI},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_propensity=True,
            supports_inference=False,  # DRLearner doesn't provide inference by default
            inference_method=None,
            requires_W=False
        )

    # ... same methods as WrappedLinearDML


class WrappedLinearDRLearner:
    """Wrapper for LinearDRLearner."""
    pass


class WrappedSparseLinearDRLearner:
    """Wrapper for SparseLinearDRLearner."""
    pass


class WrappedForestDRLearner:
    """Wrapper for ForestDRLearner."""
    pass
```

### estimators/wrappers/meta.py

```python
"""Wrappers for EconML meta-learners (S/T/X)."""

from econml.metalearners import SLearner, TLearner, XLearner
from caml.protocols.estimator import EstimatorCapabilities
from caml.data.data_schema import TreatmentType, OutcomeType


class WrappedSLearner:
    """Wrapper for EconML's S-Learner."""

    def __init__(self, **econml_kwargs):
        defaults = {"overall_model": "auto"}
        defaults.update(econml_kwargs)

        self.estimator = SLearner(**defaults)
        self._econml_kwargs = defaults

        self._capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.MULTI},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_propensity=False,
            supports_inference=False,
            requires_W=False
        )

    # ... same pattern


class WrappedTLearner:
    """Wrapper for T-Learner."""
    pass


class WrappedXLearner:
    """Wrapper for X-Learner."""
    pass
```

### estimators/wrappers/orf.py

```python
"""Wrappers for EconML Orthogonal Random Forest."""

from econml.orf import DMLOrthoForest, DROrthoForest
from caml.protocols.estimator import EstimatorCapabilities
from caml.data.data_schema import TreatmentType, OutcomeType


class WrappedDMLOrthoForest:
    """Wrapper for DMLOrthoForest."""
    pass


class WrappedDROrthoForest:
    """Wrapper for DROrthoForest."""
    pass
```

---

## 4. nuisance/

**Status**: 🔶 **NOT YET IMPLEMENTED** (directory structure exists, all files empty)

### nuisance/__init__.py

```python
"""Nuisance model estimation."""
# TODO: Import once implemented
# from caml.nuisance.tuner import NuisanceTuner
# from caml.nuisance.spec import NuisanceSpec

# __all__ = ["NuisanceTuner", "NuisanceSpec"]
```

### nuisance/spec.py

**Status**: 🔶 **NOT YET IMPLEMENTED**

Planned implementation from REFACTORING_PLAN.md:

```python
"""Nuisance specification."""
from dataclasses import dataclass


@dataclass
class NuisanceSpec:
    """Specification for nuisance models to fit."""

    fit_propensity: bool = True
    fit_outcome: bool = True
    fit_regression: bool = False

    propensity_config: dict | None = None
    outcome_config: dict | None = None
    regression_config: dict | None = None
```

### nuisance/tuner.py

See REFACTORING_PLAN.md for complete implementation - already documented in detail.

### nuisance/models.py

```python
"""Helper functions for nuisance models."""

import pandas as pd
import numpy as np


def trim_propensity(propensity: np.ndarray, bounds: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    """Trim propensity scores to avoid extreme weights."""
    return np.clip(propensity, bounds[0], bounds[1])


def prepare_features_for_propensity(data):
    """Prepare features for propensity model (X + W)."""
    if data.W is not None:
        return pd.concat([data.X, data.W], axis=1)
    return data.X


def prepare_features_for_regression(data):
    """Prepare features for regression model (X + W + T)."""
    features = [data.X]
    if data.W is not None:
        features.append(data.W)
    features.append(data.T.to_frame() if hasattr(data.T, 'to_frame') else pd.DataFrame(data.T))
    return pd.concat(features, axis=1)
```

---

## 5. scoring/

**Status**: 🔶 **NOT YET IMPLEMENTED** (directory structure exists, all files empty)

Note: The directory uses `sampling/` instead of `validation/` for cross-fitting utilities.

### scoring/__init__.py

```python
"""Scoring and evaluation metrics."""
# TODO: Import once implemented
# from caml.scoring.r_loss import RLoss
# from caml.scoring.dr_loss import DRLoss
# from caml.scoring.uplift_ import QiniScorer, AUUCScorer
# from caml.scoring.policy import PolicyValueScorer
# from caml.scoring.calibration import CalibrationScorer

# __all__ = [
#     "RLoss",
#     "DRLoss",
#     "QiniScorer",
#     "AUUCScorer",
#     "PolicyValueScorer",
#     "CalibrationScorer",
# ]
```

### scoring/base.py

**Status**: 🔶 **NOT YET IMPLEMENTED**

Planned implementation from REFACTORING_PLAN.md:

```python
"""Base scorer class."""
from abc import ABC, abstractmethod
from caml.data.dataset import CausalDataset


class BaseScorer(ABC):
    """Base class for CATE scorers."""

    @abstractmethod
    def __call__(self, estimator, data: CausalDataset) -> float:
        """Score the estimator on data.

        Parameters
        ----------
        estimator : CATEEstimator
            Estimator to score
        data : CausalDataset
            Data to score on

        Returns
        -------
        float
            Score (higher is better for Optuna)
        """
        pass
```

### scoring/r_loss.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

See REFACTORING_PLAN.md for complete RLoss implementation (Section 4.2).

### scoring/dr_loss.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

See REFACTORING_PLAN.md for complete DRLoss implementation (Section 4.3).

### scoring/uplift_.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists, note underscore suffix)

See REFACTORING_PLAN.md for complete QiniScorer implementation (Section 4.4).

Additional scorer:

```python
class AUUCScorer(QiniScorer):
    """Area Under Uplift Curve scorer."""

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Compute AUUC."""
        tau_pred = estimator.predict_cate(data.X)
        T = data.T.values if hasattr(data.T, 'values') else data.T
        Y = data.Y.values if hasattr(data.Y, 'values') else data.Y

        fractions, uplift = self.compute_uplift_curve(tau_pred, T, Y)
        auuc = np.trapz(uplift, fractions)
        return auuc

    def compute_uplift_curve(self, tau_pred, T, Y):
        """Compute uplift curve (similar to Qini but different normalization)."""
        # Implementation here
        pass
```

### scoring/policy.py

See REFACTORING_PLAN.md for complete PolicyValueScorer implementation.

### scoring/calibration.py

```python
"""CATE calibration diagnostics."""

import numpy as np
import pandas as pd
from caml.data.dataset import CausalDataset


class CalibrationScorer:
    """Check calibration of CATE predictions."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute_calibration(
        self,
        tau_pred: np.ndarray,
        tau_observed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibration by binning predicted CATE.

        Returns
        -------
        bin_centers : np.ndarray
            Center of each predicted CATE bin
        bin_observed : np.ndarray
            Average observed CATE in each bin
        """
        # Create bins
        bins = np.linspace(tau_pred.min(), tau_pred.max(), self.n_bins + 1)
        bin_indices = np.digitize(tau_pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Compute bin statistics
        bin_centers = np.zeros(self.n_bins)
        bin_observed = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_centers[i] = tau_pred[mask].mean()
                bin_observed[i] = tau_observed[mask].mean()

        return bin_centers, bin_observed

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Score calibration (return R^2 between predicted and observed)."""
        tau_pred = estimator.predict_cate(data.X)

        # Need to compute observed CATE (requires nuisance models)
        # This is a simplified version - full implementation would use DR estimates
        T = data.T.values if hasattr(data.T, 'values') else data.T
        Y = data.Y.values if hasattr(data.Y, 'values') else data.Y

        # Simplified: use raw Y differences by treatment group
        tau_observed = np.zeros_like(tau_pred)
        for i in range(len(tau_pred)):
            # This is a placeholder - real implementation needs proper CATE estimation
            pass

        bin_centers, bin_obs = self.compute_calibration(tau_pred, tau_observed)

        # Compute R^2
        ss_res = np.sum((bin_obs - bin_centers) ** 2)
        ss_tot = np.sum((bin_obs - bin_obs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return r2
```

### scoring/diagnostics.py

```python
"""Additional diagnostic metrics."""

import numpy as np
from caml.data.dataset import CausalDataset


def compute_stability(scores: list[float]) -> dict:
    """Compute stability metrics across CV folds.

    Parameters
    ----------
    scores : list[float]
        Scores from each fold

    Returns
    -------
    dict
        Stability metrics (mean, std, cv, min, max)
    """
    scores_arr = np.array(scores)
    return {
        "mean": scores_arr.mean(),
        "std": scores_arr.std(),
        "cv": scores_arr.std() / scores_arr.mean() if scores_arr.mean() != 0 else np.inf,
        "min": scores_arr.min(),
        "max": scores_arr.max(),
    }


def compute_rank_stability(rankings: list[list[str]]) -> float:
    """Compute rank stability across folds.

    Uses Kendall's tau to measure rank correlation.

    Parameters
    ----------
    rankings : list[list[str]]
        List of rankings (estimator names) from each fold

    Returns
    -------
    float
        Average Kendall's tau across all pairs of folds
    """
    from scipy.stats import kendalltau

    n_folds = len(rankings)
    correlations = []

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            # Compute Kendall's tau
            tau, _ = kendalltau(rankings[i], rankings[j])
            correlations.append(tau)

    return np.mean(correlations) if correlations else 1.0
```

---

## 6. sampling/ (formerly validation/)

**Status**: 🔶 **NOT YET IMPLEMENTED** (directory structure exists as `sampling/`, all files empty)

**Note**: The actual directory is named `sampling/` rather than `validation/` as originally planned.

### sampling/__init__.py

```python
"""Cross-fitting and validation utilities."""
# TODO: Import once implemented
# from caml.sampling.cross_fit import CrossFitter
# from caml.sampling.splitters import create_splitter
# from caml.sampling.bootstrap import BootstrapInference

# __all__ = ["CrossFitter", "create_splitter", "BootstrapInference"]
```

### sampling/cross_fit.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

See REFACTORING_PLAN.md Section 5 for complete CrossFitter implementation.

### sampling/splitters.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Planned implementation from REFACTORING_PLAN.md:

```python
"""Splitter utilities."""

from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit


def create_splitter(cv=3, groups=None, time_series=False, random_state=None):
    """Create appropriate cross-validation splitter.

    Parameters
    ----------
    cv : int
        Number of folds
    groups : array-like | None
        Group labels (for GroupKFold)
    time_series : bool
        Whether to use TimeSeriesSplit
    random_state : int | None
        Random seed

    Returns
    -------
    splitter
        sklearn splitter object
    """
    if groups is not None:
        return GroupKFold(n_splits=cv)
    elif time_series:
        return TimeSeriesSplit(n_splits=cv)
    else:
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)
```

### sampling/bootstrap.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Planned implementation from REFACTORING_PLAN.md:

```python
"""Bootstrap inference."""

import numpy as np
from sklearn.base import clone
from caml.data.dataset import CausalDataset


class BootstrapInference:
    """Bootstrap confidence intervals for CATE."""

    def __init__(self, n_bootstrap: int = 100, random_state: int | None = None):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, estimator, data: CausalDataset):
        """Fit estimator on bootstrap samples."""
        self.estimators_ = []

        rng = np.random.RandomState(self.random_state)
        n = data.n_samples

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            idx = rng.choice(n, size=n, replace=True)

            # Subset data
            data_boot = self._subset_data(data, idx)

            # Fit estimator
            est = clone(estimator)
            est.fit(data_boot)
            self.estimators_.append(est)

        return self

    def predict_interval(self, X, alpha=0.05):
        """Predict bootstrap confidence interval."""
        # Collect predictions from all bootstrap samples
        preds = np.array([est.predict_cate(X) for est in self.estimators_])

        # Compute percentiles
        lower = np.percentile(preds, 100 * alpha / 2, axis=0)
        upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def _subset_data(self, data: CausalDataset, idx):
        """Subset CausalDataset by indices."""
        # Helper function
        def _subset(arr, idx):
            if arr is None:
                return None
            if hasattr(arr, 'iloc'):
                return arr.iloc[idx]
            else:
                return arr[idx]

        return CausalDataset(
            X=_subset(data.X, idx),
            T=_subset(data.T, idx),
            Y=_subset(data.Y, idx),
            W=_subset(data.W, idx),
            treatment_type=data.treatment_type,
            outcome_type=data.outcome_type,
        )
```

---

## 7. automl/

**Status**: 🔶 **NOT YET IMPLEMENTED** (directory structure exists, all files empty)

### automl/__init__.py

```python
"""AutoCATE and AutoML components."""
# TODO: Import once implemented
# from caml.automl.auto_cate import AutoCATE

# __all__ = ["AutoCATE"]
```

### automl/auto_cate.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

See REFACTORING_PLAN.md Section 4 for planned AutoCATE implementation with Optuna.

### automl/search_space.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Planned implementation from REFACTORING_PLAN.md:

```python
"""Refactored AutoCATE with Optuna."""

import optuna
from caml.data.dataset import CausalDataset
from caml.nuisance.tuner import NuisanceTuner, NuisanceSpec
from caml.modeling.registry import get_compatible_estimators
from caml.scoring.r_loss import RLoss
from caml.scoring.dr_loss import DRLoss
from caml.validation.cross_fit import CrossFitter


class AutoCATE:
    """AutoML for CATE estimation.

    Parameters
    ----------
    nuisance_time_budget : int
        Time budget (seconds) for nuisance model tuning with FLAML
    nuisance_use_ray : bool
        Use Ray for distributed nuisance tuning
    nuisance_use_spark : bool
        Use Spark for distributed nuisance tuning
    n_trials : int
        Number of Optuna trials for CATE model selection
    scoring : str | list[str]
        Scoring metric(s): "r_loss", "dr_loss", "qini", "policy_value"
    cv : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs
    random_state : int | None
        Random seed
    verbose : int
        Verbosity level
    """

    def __init__(
        self,
        nuisance_time_budget: int = 300,
        nuisance_use_ray: bool = False,
        nuisance_use_spark: bool = False,
        n_trials: int = 100,
        scoring: str = "r_loss",
        cv: int = 3,
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: int = 2
    ):
        self.nuisance_time_budget = nuisance_time_budget
        self.nuisance_use_ray = nuisance_use_ray
        self.nuisance_use_spark = nuisance_use_spark
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.nuisance_tuner_ = None
        self.best_estimator_ = None
        self.best_score_ = None
        self.study_ = None

    def fit(
        self,
        data: CausalDataset,
        estimator_families: list[str] | None = None,
        custom_estimators: list = []
    ):
        """Fit AutoCATE.

        Parameters
        ----------
        data : CausalDataset
            Training data
        estimator_families : list[str] | None
            Estimator families to try: ["dml", "dr", "meta", "orf"]
        custom_estimators : list
            Additional custom estimators

        Returns
        -------
        self
        """
        # Validate data
        data.validate()

        # Step 1: Fit nuisance models with FLAML
        nuisance_spec = NuisanceSpec(
            fit_propensity=True,
            fit_outcome=True,
            fit_regression=False
        )

        self.nuisance_tuner_ = NuisanceTuner(
            time_budget=self.nuisance_time_budget,
            use_ray=self.nuisance_use_ray,
            use_spark=self.nuisance_use_spark,
            seed=self.random_state
        ).fit(data, nuisance_spec)

        # Step 2: Get compatible estimators
        estimators = get_compatible_estimators(
            data=data,
            families=estimator_families or ["dml", "dr", "meta"],
            custom=custom_estimators
        )

        # Step 3: Create scorer
        scorer = self._create_scorer()

        # Step 4: Optimize with Optuna
        def objective(trial):
            # Sample estimator
            idx = trial.suggest_categorical("estimator_idx", range(len(estimators)))
            estimator = estimators[idx]

            # Score it
            score = scorer(estimator, data)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        self.study_ = study
        self.best_estimator_ = estimators[study.best_params["estimator_idx"]]
        self.best_score_ = study.best_value

        # Step 5: Refit on full data
        self.best_estimator_.fit(data)

        return self

    def _create_scorer(self):
        """Create scorer based on self.scoring."""
        if self.scoring == "r_loss":
            return RLoss(
                propensity_model=self.nuisance_tuner_.propensity_model_,
                outcome_model=self.nuisance_tuner_.outcome_model_,
                cv=self.cv,
                random_state=self.random_state
            )
        elif self.scoring == "dr_loss":
            return DRLoss(
                propensity_model=self.nuisance_tuner_.propensity_model_,
                outcome_model=self.nuisance_tuner_.outcome_model_,
                cv=self.cv,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

    def predict_cate(self, X, return_interval=False, alpha=0.05):
        """Predict CATE."""
        if not return_interval:
            return self.best_estimator_.predict_cate(X)
        else:
            cate = self.best_estimator_.predict_cate(X)
            lower, upper = self.best_estimator_.predict_interval(X, alpha=alpha)
            return cate, lower, upper

    def estimate_ate(self, data: CausalDataset):
        """Estimate ATE."""
        cate = self.best_estimator_.predict_cate(data.X)
        return cate.mean()
```

### automl/search_space.py

```python
"""Optuna search space definitions."""

import optuna


class SearchSpace:
    """Define search space for AutoCATE."""

    def __init__(self, estimators: list):
        self.estimators = estimators

    def sample(self, trial: optuna.Trial):
        """Sample an estimator configuration from the search space.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object

        Returns
        -------
        estimator
            Sampled estimator
        """
        # Sample estimator type
        idx = trial.suggest_categorical("estimator_idx", range(len(self.estimators)))
        estimator = self.estimators[idx]

        # TODO: Add hyperparameter tuning per estimator
        # Example:
        # if isinstance(estimator, WrappedLinearDML):
        #     cv = trial.suggest_int("cv", 2, 5)
        #     estimator.set_params(cv=cv)

        return estimator
```

### automl/backends/__init__.py

```python
"""AutoML backends."""
```

### automl/backends/base.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Planned implementation from REFACTORING_PLAN.md:

```python
"""Base tuner backend protocol."""

from typing import Protocol


class TunerBackend(Protocol):
    """Protocol for AutoML backends."""

    def optimize(self, objective, n_trials: int, **kwargs):
        """Run optimization."""
        ...
```

### automl/backends/optuna_backend.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Planned implementation from REFACTORING_PLAN.md:

```python
"""Optuna backend implementation."""

import optuna


class OptunaBackend:
    """Optuna backend for CATE model selection."""

    def __init__(self, direction="maximize", sampler=None, **kwargs):
        self.direction = direction
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.kwargs = kwargs

    def optimize(self, objective, n_trials: int, n_jobs=1):
        """Run Optuna optimization."""
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            **self.kwargs
        )

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        return study
```

### automl/objectives.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Planned implementation from REFACTORING_PLAN.md:

```python
"""Optuna objective functions."""

from caml.scoring.r_loss import RLoss
from caml.scoring.dr_loss import DRLoss


def create_r_loss_objective(propensity_model, outcome_model, data, cv=3, random_state=None):
    """Create R-loss objective for Optuna."""
    scorer = RLoss(propensity_model, outcome_model, cv, random_state)

    def objective(trial, estimator):
        return scorer(estimator, data)

    return objective


def create_dr_loss_objective(propensity_model, outcome_model, data, cv=3, random_state=None):
    """Create DR-loss objective for Optuna."""
    scorer = DRLoss(propensity_model, outcome_model, cv, random_state)

    def objective(trial, estimator):
        return scorer(estimator, data)

    return objective
```

---

## 8. inference/

**Status**: ✅ **PARTIALLY IMPLEMENTED**

### inference/__init__.py

```python
"""Inference utilities."""
# TODO: Add bootstrap inference once implemented
# from caml.inference.results import InferenceResult
# from caml.inference.bootstrap import BootstrapInference

# __all__ = ["InferenceResult", "BootstrapInference"]
```

### inference/results.py

**Status**: ✅ **IMPLEMENTED**

```python
"""Result containers for inference."""
from dataclasses import dataclass
import numpy as np
from caml.inference.inference_schema import InferenceType


@dataclass
class InferenceResult:
    """Container for Effect inference results (can be scalar or vector of effects)."""

    point_estimate: float | np.ndarray
    stderr: float | np.ndarray | None = None
    ci_lower: float | np.ndarray | None = None
    ci_upper: float | np.ndarray | None = None
    alpha: float = 0.05
    method: InferenceType | None = None
    n_bootstrap: int | None = None  # If bootstrap was used

    def __len__(self):
        return (
            len(self.point_estimate)
            if isinstance(self.point_estimate, np.ndarray)
            else 1
        )

    def __repr__(self):
        n = len(self)
        if self.stderr is not None:
            return f"InferenceResult(n={n}, method={self.method.value if self.method else 'unknown'})"
        return f"InferenceResult(n={n})"
```

### inference/inference_schema.py

**Status**: ✅ **IMPLEMENTED**

```python
from enum import Enum


class InferenceType(Enum):
    """Categories of Supported Inference Types."""

    ANALYTIC = "analytic"
    BOOTSTRAP = "bootstrap"
```

### inference/bootstrap.py

**Status**: 🔶 **NOT YET IMPLEMENTED**

This was referenced in REFACTORING_PLAN.md but file doesn't exist yet. See `sampling/bootstrap.py` for planned implementation.

---

## 9. registry/ (formerly modeling/)

**Status**: 🔶 **NOT YET IMPLEMENTED** (directory structure exists as `registry/`, all files empty)

**Note**: The actual directory is named `registry/` rather than `modeling/` as originally planned.

### registry/__init__.py

```python
"""Model registry and discovery."""
# TODO: Import once implemented
# from caml.registry.model_bank import available_estimators
# from caml.registry.registry import get_compatible_estimators, register_estimator

# __all__ = ["available_estimators", "get_compatible_estimators", "register_estimator"]
```

### registry/model_bank.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Keep existing file, but enhance with wrappers once implemented:

```python
"""AutoCateEstimator definitions (existing + new wrappers)."""

from dataclasses import dataclass
from caml.estimators.wrappers.dml import (
    WrappedLinearDML,
    WrappedCausalForestDML,
    WrappedNonParamDML,
    WrappedSparseLinearDML,
)
from caml.estimators.wrappers.dr import (
    WrappedDRLearner,
    WrappedForestDRLearner,
    WrappedLinearDRLearner,
)
from caml.estimators.wrappers.meta import (
    WrappedSLearner,
    WrappedTLearner,
    WrappedXLearner,
)


@dataclass
class AutoCateEstimator:
    """Container for estimator with name."""
    name: str
    estimator: any


# Create wrapped estimators
AutoLinearDML = AutoCateEstimator(
    name="LinearDML",
    estimator=WrappedLinearDML()
)

AutoCausalForestDML = AutoCateEstimator(
    name="CausalForestDML",
    estimator=WrappedCausalForestDML()
)

# ... etc for all estimators

available_estimators = {
    "LinearDML": AutoLinearDML,
    "CausalForestDML": AutoCausalForestDML,
    # ... add all
}
```

### registry/registry.py

**Status**: 🔶 **NOT YET IMPLEMENTED** (empty file exists)

Planned implementation from REFACTORING_PLAN.md:

```python
"""Estimator registry and auto-discovery."""

from caml.data.dataset import CausalDataset
from caml.modeling.model_bank import available_estimators


def get_compatible_estimators(
    data: CausalDataset,
    families: list[str] | None = None,
    custom: list = []
):
    """Get estimators compatible with dataset.

    Parameters
    ----------
    data : CausalDataset
        Dataset to check compatibility
    families : list[str] | None
        Estimator families to include: ["dml", "dr", "meta", "orf"]
    custom : list
        Custom estimators to add

    Returns
    -------
    list
        Compatible estimators
    """
    # Filter by family
    if families is None:
        families = ["dml", "dr", "meta", "orf"]

    candidates = []
    for name, est_container in available_estimators.items():
        # Check if family matches
        family = None
        if "DML" in name:
            family = "dml"
        elif "DR" in name or "Learner" in name:
            family = "dr"
        elif name in ["SLearner", "TLearner", "XLearner"]:
            family = "meta"
        elif "Ortho" in name:
            family = "orf"

        if family not in families:
            continue

        # Check compatibility
        if est_container.estimator.capabilities.is_compatible(data):
            candidates.append(est_container.estimator)

    # Add custom estimators (with compatibility check)
    for est in custom:
        if est.capabilities.is_compatible(data):
            candidates.append(est)

    return candidates


def register_estimator(name: str, estimator):
    """Register a new estimator in the global registry."""
    from caml.modeling.model_bank import AutoCateEstimator
    available_estimators[name] = AutoCateEstimator(name=name, estimator=estimator)
```

---

## 10. extensions/

**Status**: ✅ **IMPLEMENTED** (New module not in original REFACTORING_PLAN)

This module was added and contains utilities for synthetic data generation and plotting.

### extensions/__init__.py

```python
"""Extensions and utilities."""
from caml.extensions.synthetic_data import SyntheticDataGenerator

__all__ = ["SyntheticDataGenerator"]
```

### extensions/synthetic_data.py

**Status**: ✅ **IMPLEMENTED** (Large file, ~48KB based on inspection)

Comprehensive synthetic data generator for causal inference testing:
- Generates flexible synthetic datasets with configurable:
  - Number and types of outcomes (continuous/binary)
  - Number and types of treatments (binary/continuous/multi-valued)
  - Number and types of effect modifiers (continuous/binary/discrete)
  - Number and types of confounders
  - Treatment effect heterogeneity patterns
  - Nonlinear relationships via GAMs
- Uses DoubleML datasets as building blocks
- Supports both linear and nonlinear data generating processes
- Includes probability truncation/renormalization for positivity
- Marked as `@experimental`

### extensions/plots.py

**Status**: ✅ **IMPLEMENTED**

Plotting utilities for causal inference (file exists, details not inspected).

---

## Summary of Implementation Status

### ✅ **FULLY IMPLEMENTED**
1. **data/** - Complete with CausalDataset, schema, validation
2. **protocols/** - Complete with CATEEstimator and InferenceProvider protocols
3. **inference/** - Partial (results.py and inference_schema.py implemented)
4. **extensions/** - Complete with SyntheticDataGenerator

### ⚠️ **PARTIALLY IMPLEMENTED**
5. **estimators/benchmark/** - InteractiveLinearRegression exists but needs protocol adaptation

### 🔶 **NOT YET IMPLEMENTED** (Structure exists, files empty)
6. **estimators/wrappers/** - All empty (dml.py, dr.py, meta.py, orf.py)
7. **nuisance/** - All empty (spec.py, tuner.py, models.py)
8. **scoring/** - All empty (r_loss.py, dr_loss.py, uplift_.py, policy.py, calibration.py, diagnostics.py)
9. **sampling/** - All empty (cross_fit.py, splitters.py, bootstrap.py)
10. **automl/** - All empty (auto_cate.py, search_space.py, objectives.py, backends/)
11. **registry/** - All empty (model_bank.py, registry.py)

### benchmarking/__init__.py

```python
"""Benchmarking utilities."""
from caml.benchmarking.harness import BenchmarkHarness

__all__ = ["BenchmarkHarness"]
```

### benchmarking/harness.py

```python
"""Benchmark harness for comparing estimators."""

import time
import numpy as np
import pandas as pd
from caml.data.dataset import CausalDataset
from caml.scoring.r_loss import RLoss


class BenchmarkHarness:
    """Compare multiple estimators on a dataset.

    Parameters
    ----------
    estimators : dict
        Dictionary of {name: estimator}
    scorers : dict
        Dictionary of {name: scorer}
    """

    def __init__(self, estimators: dict, scorers: dict):
        self.estimators = estimators
        self.scorers = scorers
        self.results_ = None

    def run(self, data: CausalDataset, n_repeats: int = 1):
        """Run benchmark.

        Parameters
        ----------
        data : CausalDataset
            Dataset to benchmark on
        n_repeats : int
            Number of repetitions

        Returns
        -------
        pd.DataFrame
            Benchmark results
        """
        results = []

        for repeat in range(n_repeats):
            for est_name, estimator in self.estimators.items():
                # Time fitting
                start = time.time()
                estimator.fit(data)
                fit_time = time.time() - start

                # Compute scores
                scores = {}
                for score_name, scorer in self.scorers.items():
                    score = scorer(estimator, data)
                    scores[score_name] = score

                # Record result
                result = {
                    "repeat": repeat,
                    "estimator": est_name,
                    "fit_time": fit_time,
                    **scores
                }
                results.append(result)

        self.results_ = pd.DataFrame(results)
        return self.results_

    def summarize(self):
        """Summarize results across repeats."""
        if self.results_ is None:
            raise ValueError("Must call .run() first")

        summary = self.results_.groupby("estimator").agg({
            "fit_time": ["mean", "std"],
            **{col: ["mean", "std"] for col in self.results_.columns if col not in ["repeat", "estimator", "fit_time"]}
        })

        return summary
```

---

---

## Directory Structure (Actual vs. Planned)

### Actual Structure
```
caml/
├── data/                    ✅ IMPLEMENTED
├── protocols/               ✅ IMPLEMENTED
├── estimators/
│   ├── benchmark/           ⚠️ PARTIAL (needs protocol adaptation)
│   └── wrappers/            🔶 EMPTY
├── nuisance/                🔶 EMPTY
├── scoring/                 🔶 EMPTY
├── sampling/                🔶 EMPTY (renamed from validation/)
├── automl/                  🔶 EMPTY
├── inference/               ⚠️ PARTIAL (results & schema done)
├── registry/                🔶 EMPTY (renamed from modeling/)
└── extensions/              ✅ IMPLEMENTED (new, not in original plan)
```

### Notable Differences from REFACTORING_PLAN.md
1. **`sampling/` vs `validation/`**: Directory renamed to `sampling/`
2. **`registry/` vs `modeling/`**: Directory renamed to `registry/`
3. **`extensions/`**: New module added with SyntheticDataGenerator and plots
4. **`effect()` vs `predict_cate()`**: Protocol uses `effect()` as method name
5. **`EstimatorCapabilities` vs `EstimatorCapabilities`**: Typo in implementation (missing 'i')
6. **`uplift_.py` vs `uplift.py`**: File has underscore suffix

---

## Next Implementation Steps (Priority Order)

Based on REFACTORING_PLAN.md Phase breakdown (UPDATED PRIORITY):

### Phase 1: ✅ COMPLETE
- CausalDataset, schema, validation, protocols

### Phase 2: 🔶 TODO - Estimator Wrappers (IMMEDIATE PRIORITY)
- [ ] Implement `estimators/wrappers/dml.py` - 4 DML wrappers
- [ ] Implement `estimators/wrappers/dr.py` - 2 DR wrappers
- [ ] Implement `estimators/wrappers/meta.py` - 3 meta-learner wrappers
- [ ] Implement `estimators/wrappers/orf.py` - 2 ORF wrappers
- [ ] Implement `registry/registry.py` - Estimator auto-discovery
- [ ] **Complete NumPy-style docstrings for all public classes/methods**
- [ ] Tests validating wrapper outputs match EconML

### Phase 3: 🔶 TODO - Nuisance Models
- [ ] Implement `nuisance/spec.py` - NuisanceSpec dataclass
- [ ] Implement `nuisance/tuner.py` - Extract NuisanceTuner from old AutoCATE
- [ ] Implement `nuisance/models.py` - Helper functions
- [ ] Refactor to use CausalDataset
- [ ] **Complete NumPy-style docstrings for all public classes/methods**
- [ ] Tests validating same outputs as old AutoCATE

### Phase 4: 🔶 TODO - Cross-Fitting & Scoring
- [ ] Implement `sampling/splitters.py` - Splitting strategies
- [ ] Implement `sampling/cross_fit.py` - CrossFitter class
- [ ] Implement `scoring/r_loss.py` - R-loss implementation
- [ ] Implement `scoring/dr_loss.py` - DR-loss implementation
- [ ] Implement `scoring/uplift_.py` - Qini, AUUC, uplift metrics
- [ ] Implement `scoring/policy.py` - Policy value scoring
- [ ] Implement `scoring/calibration.py` - Calibration metrics
- [ ] Implement `scoring/diagnostics.py` - CATE-specific diagnostics
- [ ] **Complete NumPy-style docstrings for all public classes/methods**
- [ ] Tests for scoring on synthetic data

### Phase 5: 🔶 TODO - Refactor AutoCATE
- [ ] Implement `automl/backends/base.py` - TunerBackend protocol
- [ ] Implement `automl/backends/optuna_backend.py` - Optuna implementation
- [ ] Implement `automl/objectives.py` - Optuna objective functions
- [ ] Implement `automl/search_space.py` - Search space definitions
- [ ] Refactor `automl/auto_cate.py` - Use NuisanceTuner + Optuna
- [ ] **Complete NumPy-style docstrings for all public classes/methods**
- [ ] End-to-end AutoCATE tests

### Phase 6: ⚠️ TODO - Refactor InteractiveLinearRegression
- [ ] Add `capabilities` property to InteractiveLinearRegression
- [ ] Adapt `fit()` to accept CausalDataset
- [ ] Add `effect()` method (or adapt predict)
- [ ] Ensure sklearn-compatible get_params/set_params
- [ ] Consider adding InferenceProvider protocol
- [ ] **Complete NumPy-style docstrings for all public methods**
- [ ] Ensure existing tests still pass

### Phase 7: 🔶 TODO - Testing & Documentation
- [ ] Integration tests: Full AutoCATE workflow
- [ ] Benchmarking: New vs old performance
- [ ] Migration guide
- [ ] API documentation updates
- [ ] Example notebooks
- [ ] **Final docstring audit across all modules**

---

## Critical Notes for Implementation

1. **Method naming**: Use `effect()` not `predict_cate()` per protocol
2. **Typo fix**: `EstimatorCapabilities` should be `EstimatorCapabilities`
3. **Directory names**: Use actual names (`sampling/`, `registry/`) not planned names
4. **SyntheticDataGenerator**: Already available for testing implementations
5. **InteractiveLinearRegression**: Preserve existing functionality while adding protocol compliance
