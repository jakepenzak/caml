# CaML AutoCATE Refactoring Plan

**Version:** 1.1
**Date:** January 24, 2026
**Status:** In Progress - Phase 1 Complete, Phase 2 Starting

---

## Executive Summary

**UPDATE (Jan 24, 2026)**: Phase 1 is complete! Core data structures and protocols are implemented and tested.

This plan refactors CaML into a focused **AutoCATE modeling package** with:

- **EconML-first approach**: Wrap 15-18 proven CATE estimators from EconML
- **Custom scoring infrastructure**: Build R-loss, DR-loss, Qini, and policy value evaluation from scratch
- **Dual AutoML backends**: FLAML for nuisance function tuning, Optuna for CATE model selection
- **Extracted nuisance tuner**: Reusable component for first-stage model optimization
- **Minimal custom estimators**: Only `InteractiveLinearRegression` (benchmark) for v1
- **All treatment types supported**: Binary, multi-valued, continuous from day one
- **First-class inference**: Confidence intervals and standard errors as core functionality

**Progress**: ~30% complete (Phase 1 of 7)
**Timeline**: 5-6 weeks remaining (originally 7 weeks total)
**Lines of Code**: ~500 implemented, ~3,000 remaining

---

## Table of Contents

1. [What We're Building vs. Wrapping](#1-what-were-building-vs-wrapping)
2. [Directory Structure](#2-directory-structure)
3. [Core Architecture](#3-core-architecture)
4. [Scoring Module (Custom Implementation)](#4-scoring-module-custom-implementation)
5. [Cross-Fitting Engine](#5-cross-fitting-engine)
6. [Migration Strategy](#6-migration-strategy)
7. [API Examples](#7-api-examples)
8. [Success Metrics](#8-success-metrics)

---

## 1. What We're Building vs. Wrapping

### 1.1 Wrap from EconML (~15-18 estimators)

| **Family** | **Estimators** | **Treatment Support** | **Status** |
|------------|---------------|----------------------|------------|
| **DML** | `LinearDML`, `SparseLinearDML`, `CausalForestDML`, `NonParamDML`, `KernelDML` | Binary, multi-valued, continuous | ✅ Wrap all (5) |
| **DR** | `DRLearner`, `LinearDRLearner`, `SparseLinearDRLearner`, `ForestDRLearner` | Binary, multi-valued | ✅ Wrap all (4) |
| **Meta-learners** | `SLearner`, `TLearner`, `XLearner` | Binary, multi-valued | ✅ Wrap all (3) |
| **ORF** | `DMLOrthoForest`, `DROrthoForest` | Binary, continuous | ✅ Wrap both (2) |
| **IV** | `DMLIV`, `DRIV`, etc. | With instruments | 🔶 v2 |
| **Panel** | `DynamicDML` | Time-series | 🔶 v2 |

**Total for v1**: 14 wrapped estimators

### 1.2 Build Custom (Minimal Set)

| **Component** | **Why Custom?** | **Priority** |
|--------------|----------------|--------------|
| **InteractiveLinearRegression** | Existing benchmark, formula-based OLS | ✅ **v1** - Refactor to protocol |
| **Scoring Infrastructure** | Full control over R-loss, DR-loss, Qini, policy value | ✅ **v1** - Build from scratch |
| **Cross-Fitting Engine** | Need for custom scoring, flexible splitters | ✅ **v1** - Build from scratch |
| **NuisanceTuner** | Extract from AutoCATE, make reusable | ✅ **v1** - Extract & refactor |
| **Honest Causal Forest** | EconML's lacks true honesty + variance | 🔶 **v2** |
| **Quantile CATE** | Not in EconML | 🔶 **v2** |

### 1.3 Scoring Components (All Custom)

| **Metric** | **Implementation** | **Priority** |
|-----------|-------------------|--------------|
| **R-loss** | Custom orthogonal score with cross-fitted nuisances | ✅ **v1** |
| **DR-loss** | Custom doubly-robust pseudo-outcome loss | ✅ **v1** |
| **Qini coefficient / curve** | Custom uplift metric for binary treatment | ✅ **v1** |
| **AUUC** | Area under uplift curve | ✅ **v1** |
| **Policy value (DR)** | Doubly-robust policy evaluation | ✅ **v1** |
| **Calibration** | CATE calibration by deciles | ✅ **v1** |
| **Stability diagnostics** | Cross-fold variance, rank stability | ✅ **v1** |

**Rationale**: Building scoring ourselves provides:
- Full control over implementation details
- Flexibility to add new metrics
- No dependency on EconML's validation API
- Ability to optimize for performance
- Custom diagnostics tailored to our use cases

---

## 2. Directory Structure

```
caml/
├── __init__.py
├── _version.py
├── README.md
├── REFACTORING_PLAN.md          # This document
│
├── data/                         # ✅ IMPLEMENTED - Data containers & validation
│   ├── __init__.py
│   ├── dataset.py                # ✅ CausalDataset class
│   ├── data_schema.py            # ✅ TreatmentType, OutcomeType, Estimand enums
│   └── _validation.py            # ✅ Overlap, positivity, missing data checks
│
├── protocols/                    # ✅ IMPLEMENTED - Core interfaces
│   ├── __init__.py
│   ├── estimator.py              # ✅ CATEEstimator Protocol + EstimatorCapabilities
│   └── inference.py              # ✅ InferenceProvider Protocol
│
├── estimators/                   # ⚠️ PARTIAL - CATE estimators
│   ├── __init__.py
│   ├── base.py                   # 🔶 Empty (optional mixin, low priority)
│   ├── benchmark/                # ⚠️ Needs refactoring
│   │   ├── __init__.py
│   │   └── interactive_ols.py    # ⚠️ EXISTS - Needs protocol adaptation
│   └── wrappers/                 # 🔶 TODO - EconML wrappers
│       ├── __init__.py           # 🔶 Empty
│       ├── dml.py                # 🔶 Empty - Wrap LinearDML, CausalForestDML, etc.
│       ├── dr.py                 # 🔶 Empty - Wrap DRLearner, ForestDRLearner, etc.
│       ├── meta.py               # 🔶 Empty - Wrap SLearner, TLearner, XLearner
│       └── orf.py                # 🔶 Empty - Wrap DMLOrthoForest, DROrthoForest
│
├── nuisance/                     # 🔶 TODO - First-stage nuisance estimation
│   ├── __init__.py               # 🔶 Empty
│   ├── tuner.py                  # 🔶 Empty - NuisanceTuner (FLAML-based)
│   ├── spec.py                   # 🔶 Empty - NuisanceSpec dataclass
│   └── models.py                 # 🔶 Empty - Helper functions
│
├── scoring/                      # 🔶 TODO - Scoring & evaluation (ALL CUSTOM)
│   ├── __init__.py               # 🔶 Empty
│   ├── base.py                   # 🔶 Empty - BaseScorer abstract class
│   ├── r_loss.py                 # 🔶 Empty - R-learner loss (orthogonal score)
│   ├── dr_loss.py                # 🔶 Empty - Doubly-robust loss
│   ├── uplift_.py                # 🔶 Empty - Qini, AUUC, uplift curves (note underscore)
│   ├── policy.py                 # 🔶 Empty - Policy value (IPS, DR policy evaluation)
│   ├── calibration.py            # 🔶 Empty - CATE calibration diagnostics
│   └── diagnostics.py            # 🔶 Empty - Stability, sensitivity, overlap checks
│
├── sampling/                     # 🔶 TODO - Cross-fitting & resampling (renamed from validation/)
│   ├── __init__.py               # 🔶 Empty
│   ├── cross_fit.py              # 🔶 Empty - CrossFitter class (core engine)
│   ├── splitters.py              # 🔶 Empty - KFold, GroupKFold, TimeSeriesSplit wrappers
│   └── bootstrap.py              # 🔶 Empty - Bootstrap inference wrapper
│
├── automl/                       # 🔶 TODO - AutoCATE orchestration
│   ├── __init__.py               # 🔶 Empty
│   ├── auto_cate.py              # 🔶 Empty - Main AutoCATE class
│   ├── search_space.py           # 🔶 Empty - Optuna search space definitions
│   ├── backends/
│   │   ├── __init__.py           # 🔶 Empty
│   │   ├── base.py               # 🔶 Empty - TunerBackend Protocol
│   │   └── optuna_backend.py     # 🔶 Empty - Optuna implementation for CATE selection
│   └── objectives.py             # 🔶 Empty - Optuna objectives (R-loss, DR-loss, multi-metric)
│
├── inference/                    # ⚠️ PARTIAL - Inference utilities
│   ├── __init__.py               # 🔶 Empty
│   ├── results.py                # ✅ InferenceResult dataclass
│   └── inference_schema.py                 # ✅ InferenceType enum
│
├── registry/                     # 🔶 TODO - Model registry (renamed from modeling/)
│   ├── __init__.py               # 🔶 Empty
│   ├── model_bank.py             # 🔶 Empty - AutoCateEstimator definitions
│   └── registry.py               # 🔶 Empty - Auto-discovery of wrapped estimators
│
├── extensions/                   # ✅ IMPLEMENTED - New utilities (not in original plan)
│   ├── __init__.py               # ✅ Implemented
│   ├── synthetic_data.py         # ✅ SyntheticDataGenerator
│   └── plots.py                  # ✅ Plotting utilities
│
├── _generics/                    # ✅ EXISTING - Utilities (keep)
│   ├── logging.py
│   ├── decorators.py
│   └── utils.py
│
└── (other existing modules preserved for compatibility)
```

**Legend**:
- ✅ **IMPLEMENTED** - Complete and tested
- ⚠️ **PARTIAL** - Exists but needs work
- 🔶 **TODO** - Not yet implemented (may have empty file)

---

## 3. Core Architecture

### 3.1 CausalDataset (data/dataset.py)

**Purpose**: Unified data container with metadata and validation

```python
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from caml.data.data_schema import TreatmentType, OutcomeType

@dataclass
class CausalDataset:
    """Unified causal data container."""

    # Core data (always required)
    X: pd.DataFrame | np.ndarray      # Effect modifiers
    T: pd.Series | np.ndarray         # Treatment
    Y: pd.Series | np.ndarray         # Outcome

    # Optional data
    W: pd.DataFrame | np.ndarray | None = None  # Controls (separate from X)
    weights: np.ndarray | None = None
    propensity: np.ndarray | None = None  # Pre-computed propensity scores

    # Grouping/clustering
    cluster_id: np.ndarray | None = None
    strata: np.ndarray | None = None

    # Metadata
    treatment_type: TreatmentType = field(default=TreatmentType.BINARY)
    outcome_type: OutcomeType = field(default=OutcomeType.CONTINUOUS)

    # Feature names
    X_names: list[str] | None = None
    W_names: list[str] | None = None
    T_name: str = "treatment"
    Y_name: str = "outcome"

    def validate(self) -> None:
        """Run validation checks (shapes, types, overlap, missing data)."""
        # Check shapes match
        # Check for missing values
        # Check treatment type matches data
        # Check overlap/positivity
        pass

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        X: list[str],
        T: str,
        Y: str,
        W: list[str] | None = None,
        treatment_type: TreatmentType = TreatmentType.BINARY,
        outcome_type: OutcomeType = OutcomeType.CONTINUOUS,
        **kwargs
    ) -> "CausalDataset":
        """Construct from pandas DataFrame."""
        return cls(
            X=df[X],
            T=df[T],
            Y=df[Y],
            W=df[W] if W else None,
            treatment_type=treatment_type,
            outcome_type=outcome_type,
            X_names=X,
            W_names=W,
            T_name=T,
            Y_name=Y,
            **kwargs
        )
```

### 3.2 Estimator Protocol (protocols/estimator.py)

**Purpose**: Define interface all estimators must satisfy

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
from caml.data.dataset import CausalDataset
from caml.data.data_schema import TreatmentType, OutcomeType

@dataclass(frozen=True)
class EstimatorCapabilities:
    """Describes what an estimator supports."""
    treatment_types: set[TreatmentType]
    outcome_types: set[OutcomeType]
    requires_propensity: bool = False
    accepts_given_propensity: bool = False
    supports_inference: bool = False
    inference_method: str | None = None  # "analytic", "bootstrap", "both"
    requires_W: bool = False  # Needs separate confounders

@runtime_checkable
class CATEEstimator(Protocol):
    """Core protocol for CATE estimators."""

    capabilities: EstimatorCapabilities

    def fit(self, data: CausalDataset, **kwargs) -> "CATEEstimator":
        """Fit the estimator."""
        ...

    def predict_cate(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict CATE for given features."""
        ...

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters (sklearn compatibility)."""
        ...

    def set_params(self, **params) -> "CATEEstimator":
        """Set parameters (sklearn compatibility)."""
        ...
```

### 3.3 Estimator Wrapper Pattern (estimators/wrappers/dml.py)

**Purpose**: Wrap EconML estimators to satisfy our protocol

```python
from econml.dml import LinearDML
from caml.protocols.estimator import CATEEstimator, EstimatorCapabilities
from caml.data.data_schema import TreatmentType, OutcomeType
from caml.data.dataset import CausalDataset
import numpy as np

class WrappedLinearDML:
    """Wrapper for EconML's LinearDML."""

    def __init__(self, **econml_kwargs):
        self.estimator = LinearDML(**econml_kwargs)
        self._capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.MULTI, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_propensity=False,
            supports_inference=True,
            inference_method="analytic",
            requires_W=True
        )

    @property
    def capabilities(self) -> EstimatorCapabilities:
        return self._capabilities

    def fit(self, data: CausalDataset, **kwargs) -> "WrappedLinearDML":
        """Fit using EconML's API."""
        self.estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X is not None else None,
            W=data.W if data.W is not None else None,
            **kwargs
        )
        return self

    def predict_cate(self, X, **kwargs) -> np.ndarray:
        """Map to EconML's effect()."""
        return self.estimator.effect(X, **kwargs)

    def get_params(self, deep=True) -> dict:
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params) -> "WrappedLinearDML":
        self.estimator.set_params(**params)
        return self
```

### 3.4 Nuisance Tuner (nuisance/tuner.py)

**Purpose**: Extract nuisance model tuning from AutoCATE

```python
from dataclasses import dataclass
from flaml import AutoML
from caml.data.dataset import CausalDataset
import pandas as pd

@dataclass
class NuisanceSpec:
    """Specification for which nuisance models to fit."""
    fit_propensity: bool = True       # E[T|X,W]
    fit_outcome: bool = True           # E[Y|X,W]
    fit_regression: bool = False       # E[Y|X,W,T] for DR methods

    # FLAML config overrides
    propensity_config: dict | None = None
    outcome_config: dict | None = None
    regression_config: dict | None = None

class NuisanceTuner:
    """FLAML-based nuisance model tuner (extracted from AutoCATE)."""

    def __init__(
        self,
        time_budget: int = 300,
        use_ray: bool = False,
        use_spark: bool = False,
        seed: int | None = None,
        verbose: int = 0
    ):
        self.time_budget = time_budget
        self.use_ray = use_ray
        self.use_spark = use_spark
        self.seed = seed
        self.verbose = verbose

        # Fitted models (stored after fit())
        self.propensity_model_ = None
        self.outcome_model_ = None
        self.regression_model_ = None

    def fit(self, data: CausalDataset, spec: NuisanceSpec) -> "NuisanceTuner":
        """Fit nuisance models based on spec."""

        base_config = self._build_base_config()

        if spec.fit_propensity:
            config = self._build_propensity_config(data, base_config, spec)
            self.propensity_model_ = self._run_flaml(config)

        if spec.fit_outcome:
            config = self._build_outcome_config(data, base_config, spec)
            self.outcome_model_ = self._run_flaml(config)

        if spec.fit_regression:
            config = self._build_regression_config(data, base_config, spec)
            self.regression_model_ = self._run_flaml(config)

        return self

    def _build_base_config(self) -> dict:
        """Base FLAML settings."""
        config = {
            "n_jobs": -1,
            "time_budget": self.time_budget,
            "seed": self.seed,
            "verbose": self.verbose,
            "log_file_name": "",
            "early_stop": True,
            "eval_method": "cv",
            "n_splits": 3,
        }

        if self.use_ray:
            config["use_ray"] = True
            config["n_concurrent_trials"] = 4
        elif self.use_spark:
            config["use_spark"] = True
            config["n_concurrent_trials"] = 4

        return config

    def _build_propensity_config(self, data: CausalDataset, base: dict, spec: NuisanceSpec) -> dict:
        """Build config for propensity model E[T|X,W]."""
        config = base.copy()

        # Determine task type
        if data.treatment_type.is_discrete():
            config["task"] = "classification"
            config["metric"] = "log_loss"
        else:
            config["task"] = "regression"
            config["metric"] = "mse"

        # Prepare data
        XW = pd.concat([data.X, data.W], axis=1) if data.W is not None else data.X
        config["X_train"] = XW
        config["y_train"] = data.T

        # Apply user overrides
        if spec.propensity_config:
            config.update(spec.propensity_config)

        return config

    def _build_outcome_config(self, data: CausalDataset, base: dict, spec: NuisanceSpec) -> dict:
        """Build config for outcome model E[Y|X,W]."""
        config = base.copy()

        # Determine task type
        if data.outcome_type == OutcomeType.BINARY:
            config["task"] = "classification"
            config["metric"] = "log_loss"
        else:
            config["task"] = "regression"
            config["metric"] = "mse"

        # Prepare data
        XW = pd.concat([data.X, data.W], axis=1) if data.W is not None else data.X
        config["X_train"] = XW
        config["y_train"] = data.Y

        # Apply user overrides
        if spec.outcome_config:
            config.update(spec.outcome_config)

        return config

    def _build_regression_config(self, data: CausalDataset, base: dict, spec: NuisanceSpec) -> dict:
        """Build config for regression model E[Y|X,W,T]."""
        config = base.copy()

        # Task type same as outcome
        if data.outcome_type == OutcomeType.BINARY:
            config["task"] = "classification"
            config["metric"] = "log_loss"
        else:
            config["task"] = "regression"
            config["metric"] = "mse"

        # Prepare data (include treatment)
        XWT = pd.concat([data.X, data.W, data.T], axis=1) if data.W is not None else pd.concat([data.X, data.T], axis=1)
        config["X_train"] = XWT
        config["y_train"] = data.Y

        # Apply user overrides
        if spec.regression_config:
            config.update(spec.regression_config)

        return config

    def _run_flaml(self, config: dict):
        """Run AutoML and return best estimator."""
        automl = AutoML()
        automl.fit(**config)
        return automl.model.estimator
```

---

## 4. Scoring Module (Custom Implementation)

**Critical Component**: All scoring built from scratch for full control

### 4.1 Cross-Fitting Primer

For unbiased CATE model selection, we need **orthogonal scores** computed with out-of-fold nuisance predictions:

1. Split data into K folds
2. For each fold k:
   - Fit nuisance models on data excluding fold k
   - Predict nuisances on fold k (out-of-fold)
3. Compute score using out-of-fold predictions

### 4.2 R-Loss (scoring/r_loss.py)

**Definition**: R-learner objective for CATE model selection

$$
R\text{-loss} = \mathbb{E}\left[(Y - \hat{m}(X) - \tau(X) \cdot (T - \hat{e}(X)))^2\right]
$$

Where:
- $\hat{m}(X) = \mathbb{E}[Y|X,W]$ (outcome model)
- $\hat{e}(X) = \mathbb{E}[T|X,W]$ (propensity model)
- $\tau(X)$ is the CATE model being scored

```python
import numpy as np
from sklearn.base import clone
from caml.data.dataset import CausalDataset
from caml.validation.cross_fit import CrossFitter

class RLoss:
    """R-learner loss for CATE model selection."""

    def __init__(
        self,
        propensity_model,
        outcome_model,
        cv: int = 3,
        random_state: int | None = None
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.cv = cv
        self.random_state = random_state
        self._cross_fitter = CrossFitter(cv=cv, random_state=random_state)

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Compute out-of-fold R-loss."""

        # Step 1: Get out-of-fold nuisance predictions
        m_hat, e_hat = self._cross_fitter.fit_predict_nuisances_dml(
            data=data,
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model
        )

        # Step 2: Compute residuals
        Y_res = data.Y - m_hat  # Outcome residual
        T_res = data.T - e_hat  # Treatment residual

        # Step 3: Fit estimator on full data and predict CATE
        estimator_fitted = clone(estimator).fit(data)
        tau_hat = estimator_fitted.predict_cate(data.X)

        # Step 4: Compute R-loss
        squared_error = (Y_res - tau_hat * T_res) ** 2
        r_loss = np.mean(squared_error)

        # Return negative (Optuna maximizes, we want to minimize loss)
        return -r_loss
```

### 4.3 DR-Loss (scoring/dr_loss.py)

**Definition**: Doubly-robust pseudo-outcome loss

$$
\text{DR-loss} = \mathbb{E}\left[(\phi_{DR}(X) - \tau(X))^2\right]
$$

Where $\phi_{DR}$ is the doubly-robust pseudo-outcome:

$$
\phi_{DR}(X) = \frac{T \cdot Y}{e(X)} - \frac{(1-T) \cdot Y}{1-e(X)} + \left(\frac{T}{e(X)} - \frac{1-T}{1-e(X)}\right) \hat{\mu}(T,X)
$$

```python
import numpy as np
from sklearn.base import clone
from caml.data.dataset import CausalDataset
from caml.validation.cross_fit import CrossFitter

class DRLoss:
    """Doubly-robust loss for CATE model selection."""

    def __init__(
        self,
        propensity_model,
        outcome_model,
        cv: int = 3,
        trim_propensity: tuple[float, float] = (0.01, 0.99),
        random_state: int | None = None
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.cv = cv
        self.trim_propensity = trim_propensity
        self.random_state = random_state
        self._cross_fitter = CrossFitter(cv=cv, random_state=random_state)

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Compute out-of-fold DR-loss."""

        # Assumes binary treatment for v1
        if data.treatment_type != TreatmentType.BINARY:
            raise ValueError("DRLoss currently only supports binary treatment")

        # Step 1: Get out-of-fold nuisance predictions
        mu_0, mu_1, e_hat = self._cross_fitter.fit_predict_nuisances_dr(
            data=data,
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model
        )

        # Step 2: Trim propensity scores (avoid extreme weights)
        e_hat_trimmed = np.clip(e_hat, self.trim_propensity[0], self.trim_propensity[1])

        # Step 3: Compute DR pseudo-outcome
        T = data.T.values if hasattr(data.T, 'values') else data.T
        Y = data.Y.values if hasattr(data.Y, 'values') else data.Y

        phi_dr = (
            (T * Y) / e_hat_trimmed - ((1 - T) * Y) / (1 - e_hat_trimmed)
            + (1 - T / e_hat_trimmed) * mu_1
            - (1 - (1 - T) / (1 - e_hat_trimmed)) * mu_0
        )

        # Step 4: Fit estimator and predict CATE
        estimator_fitted = clone(estimator).fit(data)
        tau_hat = estimator_fitted.predict_cate(data.X)

        # Step 5: Compute DR-loss
        squared_error = (phi_dr - tau_hat) ** 2
        dr_loss = np.mean(squared_error)

        return -dr_loss
```

### 4.4 Qini & Uplift (scoring/uplift.py)

**Purpose**: Evaluate targeting/policy performance

```python
import numpy as np
from caml.data.dataset import CausalDataset

class QiniScorer:
    """Qini coefficient and curve for uplift evaluation."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute_qini_curve(
        self,
        tau_pred: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Qini curve.

        Returns
        -------
        fractions : np.ndarray
            Fraction of population targeted (0 to 1)
        qini : np.ndarray
            Cumulative uplift at each fraction
        """
        # Sort by predicted CATE (descending)
        order = np.argsort(-tau_pred)
        T_sorted = T[order]
        Y_sorted = Y[order]

        n = len(T_sorted)
        fractions = np.linspace(0, 1, self.n_bins + 1)
        qini = np.zeros(self.n_bins + 1)

        for i, frac in enumerate(fractions):
            n_targeted = int(frac * n)
            if n_targeted == 0:
                continue

            T_targeted = T_sorted[:n_targeted]
            Y_targeted = Y_sorted[:n_targeted]

            # Compute uplift for targeted group
            n_treated = T_targeted.sum()
            n_control = n_targeted - n_treated

            if n_treated > 0 and n_control > 0:
                y_treated = Y_targeted[T_targeted == 1].mean()
                y_control = Y_targeted[T_targeted == 0].mean()
                uplift_per_unit = y_treated - y_control
                qini[i] = uplift_per_unit * n_targeted

        return fractions, qini

    def compute_qini_coefficient(
        self,
        tau_pred: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """Compute Qini coefficient (area between Qini curve and diagonal)."""
        fractions, qini = self.compute_qini_curve(tau_pred, T, Y)

        # Area under Qini curve
        auc_qini = np.trapz(qini, fractions)

        # Area under random targeting (diagonal)
        random_qini = qini[-1] * 0.5  # Triangle area

        # Qini coefficient
        return (auc_qini - random_qini) / random_qini if random_qini != 0 else 0.0

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Score estimator using Qini coefficient."""
        tau_pred = estimator.predict_cate(data.X)
        T = data.T.values if hasattr(data.T, 'values') else data.T
        Y = data.Y.values if hasattr(data.Y, 'values') else data.Y

        return self.compute_qini_coefficient(tau_pred, T, Y)
```

### 4.5 Policy Value (scoring/policy.py)

**Purpose**: Evaluate expected value of treatment assignment policy

```python
import numpy as np
from caml.data.dataset import CausalDataset

class PolicyValueScorer:
    """Doubly-robust policy value estimation."""

    def __init__(
        self,
        propensity_model,
        outcome_model,
        policy_fn=None,  # Custom policy function
        trim_propensity: tuple[float, float] = (0.01, 0.99)
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.policy_fn = policy_fn
        self.trim_propensity = trim_propensity

    def __call__(self, estimator, data: CausalDataset, threshold: float = 0.0) -> float:
        """Compute DR policy value.

        Policy: treat if predicted CATE > threshold
        """
        # Predict CATE
        tau_pred = estimator.predict_cate(data.X)

        # Define policy (default: treat if positive effect)
        if self.policy_fn is None:
            pi = (tau_pred > threshold).astype(float)
        else:
            pi = self.policy_fn(tau_pred)

        # Get nuisance predictions
        e_hat = self.propensity_model.predict_proba(
            pd.concat([data.X, data.W], axis=1) if data.W is not None else data.X
        )[:, 1]

        mu_1 = self.outcome_model.predict(
            pd.concat([data.X, data.W, pd.Series(np.ones(len(data.X)))], axis=1)
        )
        mu_0 = self.outcome_model.predict(
            pd.concat([data.X, data.W, pd.Series(np.zeros(len(data.X)))], axis=1)
        )

        # Trim propensity
        e_hat_trimmed = np.clip(e_hat, self.trim_propensity[0], self.trim_propensity[1])

        # DR policy value
        T = data.T.values if hasattr(data.T, 'values') else data.T
        Y = data.Y.values if hasattr(data.Y, 'values') else data.Y

        value = np.mean(
            pi * (T * Y / e_hat_trimmed - (T - e_hat_trimmed) / e_hat_trimmed * mu_1)
            + (1 - pi) * ((1 - T) * Y / (1 - e_hat_trimmed) + (T - e_hat_trimmed) / (1 - e_hat_trimmed) * mu_0)
        )

        return value
```

---

## 5. Cross-Fitting Engine

**Purpose**: Provide out-of-fold predictions for unbiased scoring

### validation/cross_fit.py

```python
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.base import clone
from caml.data.dataset import CausalDataset
import pandas as pd

class CrossFitter:
    """Cross-fitting engine for orthogonal scores."""

    def __init__(
        self,
        cv: int = 3,
        random_state: int | None = None,
        group_col: str | None = None
    ):
        self.cv = cv
        self.random_state = random_state
        self.group_col = group_col

    def fit_predict_nuisances_dml(
        self,
        data: CausalDataset,
        outcome_model,
        propensity_model
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit nuisance models with cross-fitting and return out-of-fold predictions.

        Returns
        -------
        m_hat : np.ndarray
            Out-of-fold outcome predictions E[Y|X,W]
        e_hat : np.ndarray
            Out-of-fold propensity predictions E[T|X,W]
        """
        n = len(data.Y)
        m_hat = np.zeros(n)
        e_hat = np.zeros(n)

        # Create splitter
        if self.group_col:
            splitter = GroupKFold(n_splits=self.cv)
            groups = data.X[self.group_col] if isinstance(data.X, pd.DataFrame) else None
        else:
            splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            groups = None

        # Prepare features
        XW = pd.concat([data.X, data.W], axis=1) if data.W is not None else data.X

        # Cross-fit
        for train_idx, val_idx in splitter.split(XW, groups=groups):
            # Split data
            XW_train, XW_val = XW.iloc[train_idx], XW.iloc[val_idx]
            Y_train = data.Y.iloc[train_idx] if hasattr(data.Y, 'iloc') else data.Y[train_idx]
            T_train = data.T.iloc[train_idx] if hasattr(data.T, 'iloc') else data.T[train_idx]

            # Fit and predict outcome model
            m_model = clone(outcome_model)
            m_model.fit(XW_train, Y_train)
            m_hat[val_idx] = m_model.predict(XW_val)

            # Fit and predict propensity model
            e_model = clone(propensity_model)
            e_model.fit(XW_train, T_train)

            if hasattr(e_model, 'predict_proba'):
                e_hat[val_idx] = e_model.predict_proba(XW_val)[:, 1]
            else:
                e_hat[val_idx] = e_model.predict(XW_val)

        return m_hat, e_hat

    def fit_predict_nuisances_dr(
        self,
        data: CausalDataset,
        outcome_model,
        propensity_model
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit nuisance models for DR-loss (need mu_0, mu_1, e).

        Returns
        -------
        mu_0 : np.ndarray
            E[Y|X,W,T=0]
        mu_1 : np.ndarray
            E[Y|X,W,T=1]
        e_hat : np.ndarray
            E[T|X,W]
        """
        n = len(data.Y)
        mu_0 = np.zeros(n)
        mu_1 = np.zeros(n)
        e_hat = np.zeros(n)

        # Create splitter
        if self.group_col:
            splitter = GroupKFold(n_splits=self.cv)
            groups = data.X[self.group_col] if isinstance(data.X, pd.DataFrame) else None
        else:
            splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            groups = None

        # Prepare features
        XW = pd.concat([data.X, data.W], axis=1) if data.W is not None else data.X

        # Cross-fit
        for train_idx, val_idx in splitter.split(XW, groups=groups):
            XW_train, XW_val = XW.iloc[train_idx], XW.iloc[val_idx]
            Y_train = data.Y.iloc[train_idx] if hasattr(data.Y, 'iloc') else data.Y[train_idx]
            T_train = data.T.iloc[train_idx] if hasattr(data.T, 'iloc') else data.T[train_idx]

            # Fit propensity
            e_model = clone(propensity_model)
            e_model.fit(XW_train, T_train)
            e_hat[val_idx] = e_model.predict_proba(XW_val)[:, 1]

            # Fit outcome models (conditional on treatment)
            # mu_1: E[Y|X,W,T=1]
            XWT_train_1 = pd.concat([XW_train, pd.Series(np.ones(len(XW_train)))], axis=1)
            XWT_val_1 = pd.concat([XW_val, pd.Series(np.ones(len(XW_val)))], axis=1)
            m1_model = clone(outcome_model)
            m1_model.fit(XWT_train_1, Y_train)
            mu_1[val_idx] = m1_model.predict(XWT_val_1)

            # mu_0: E[Y|X,W,T=0]
            XWT_train_0 = pd.concat([XW_train, pd.Series(np.zeros(len(XW_train)))], axis=1)
            XWT_val_0 = pd.concat([XW_val, pd.Series(np.zeros(len(XW_val)))], axis=1)
            m0_model = clone(outcome_model)
            m0_model.fit(XWT_train_0, Y_train)
            mu_0[val_idx] = m0_model.predict(XWT_val_0)

        return mu_0, mu_1, e_hat
```

---

## 6. Migration Strategy

### Phase 1: Foundation (Week 1) ✅ **COMPLETE**
**Goal**: Core data structures and protocols

1. ✅ Create `data/dataset.py` - `CausalDataset` class
2. ✅ Create `data/data_schema.py` - `TreatmentType`, `OutcomeType`, `Estimand` enums
3. ✅ Create `data/_validation.py` - validation functions
4. ✅ Create `protocols/estimator.py` - `CATEEstimator` protocol, `EstimatorCapabilities`
5. ✅ Create `inference/results.py` - `InferenceResult` dataclass
6. ✅ Create `inference/inference_schema.py` - `InferenceType` enum
7. ✅ Create `protocols/inference.py` - `InferenceProvider` protocol
8. ✅ **Tests**: Validate `CausalDataset.from_dataframe()`, validation logic

**Deliverable**: ✅ Working `CausalDataset` with comprehensive tests

**Notes**:
- Implementation uses `EstimatorCapabilities` (typo - missing 'i')
- Protocol uses `effect()` method name instead of `predict_cate()`
- Bonus: `extensions/` module added with `SyntheticDataGenerator` for testing

---

### Phase 2: Estimator Wrappers (Week 2) 🔶 **IN PROGRESS**
**Goal**: Wrap EconML estimators

7. 🔶 Create `estimators/wrappers/dml.py` - Wrap 4 DML estimators
8. 🔶 Create `estimators/wrappers/dr.py` - Wrap 2 DR estimators
9. 🔶 Create `estimators/wrappers/meta.py` - Wrap S/T/X learners
10. 🔶 Create `estimators/wrappers/orf.py` - Wrap ORF estimators
11. 🔶 Create `registry/registry.py` - Estimator auto-discovery (note: directory renamed from `modeling/`)
12. 🔶 **Docstrings**: Complete NumPy-style docstrings for all public classes/methods with runnable examples
13. 🔶 **Tests**: Validate wrapper outputs match EconML directly

**Deliverable**: 14 wrapped estimators with complete docstrings and tests

**Implementation Notes**:
- Directory is `registry/` not `modeling/`
- Each wrapper must implement `CATEEstimator` protocol
- Use `effect()` method name (not `predict_cate()`)
- Set `capabilities` property correctly
- Follow wrapper pattern from REFACTORING_PLAN.md Section 3.3
- **CRITICAL**: Docstrings are mandatory before marking phase complete

---

### Phase 3: Nuisance Models (Week 3) 🔶 **TODO**
**Goal**: Extract from AutoCATE

14. 🔶 Create `nuisance/spec.py` - `NuisanceSpec` dataclass
15. 🔶 Create `nuisance/tuner.py` - Extract `NuisanceTuner` from old `AutoCATE._find_nuisance_functions()`
16. 🔶 Create `nuisance/models.py` - Helper functions for nuisance models
17. 🔶 Refactor to use `CausalDataset`
18. 🔶 **Docstrings**: Complete NumPy-style docstrings for all public classes/methods with runnable examples
19. 🔶 **Tests**: Validate `NuisanceTuner` produces same models as old AutoCATE

**Deliverable**: Standalone `NuisanceTuner` with complete docstrings and tests

**Implementation Notes**:
- Look for existing AutoCATE implementation to extract logic
- Follow REFACTORING_PLAN.md Section 3.4 for detailed implementation
- **CRITICAL**: Docstrings are mandatory before marking phase complete

---

### Phase 4: Cross-Fitting & Scoring (Week 4) 🔶 **TODO**
**Goal**: Custom scoring infrastructure

20. 🔶 Create `sampling/splitters.py` - Splitting strategies
21. 🔶 Create `sampling/cross_fit.py` - `CrossFitter` class (note: directory renamed from `validation/`)
22. 🔶 Create `scoring/r_loss.py` - `RLoss` scorer
23. 🔶 Create `scoring/dr_loss.py` - `DRLoss` scorer
24. 🔶 Create `scoring/uplift_.py` - `QiniScorer` (note: underscore suffix in filename)
25. 🔶 Create `scoring/policy.py` - `PolicyValueScorer`
26. 🔶 Create `scoring/calibration.py` - Calibration metrics
27. 🔶 Create `scoring/diagnostics.py` - CATE diagnostics
28. 🔶 **Docstrings**: Complete NumPy-style docstrings for all public classes/methods with runnable examples
29. 🔶 **Tests**: Validate scoring on synthetic data with known ground truth (use `SyntheticDataGenerator`)

**Deliverable**: Working scoring module with complete docstrings and tests

**Implementation Notes**:
- Directory is `sampling/` not `validation/`
- File is `uplift_.py` (with underscore) not `uplift.py`
- Use `SyntheticDataGenerator` from `extensions/` for test data
- Follow examples in REFACTORING_PLAN.md Sections 4.2-4.5
- **CRITICAL**: Docstrings are mandatory before marking phase complete
- **Dependencies**: Phase 3 complete (nuisance models needed for DR-loss)

---

### Phase 5: Refactor AutoCATE (Week 5) 🔶 **TODO**
**Goal**: New AutoCATE with Optuna

30. 🔶 Create `automl/backends/base.py` - `TunerBackend` protocol
31. 🔶 Create `automl/backends/optuna_backend.py` - Optuna implementation
32. 🔶 Create `automl/objectives.py` - Optuna objective functions
33. 🔶 Create `automl/search_space.py` - Search space definitions
34. 🔶 Refactor `automl/auto_cate.py` - Use `NuisanceTuner` + Optuna + registry
35. 🔶 **Docstrings**: Complete NumPy-style docstrings for all public classes/methods with runnable examples
36. 🔶 **Tests**: End-to-end AutoCATE on synthetic data

**Deliverable**: New AutoCATE with complete docstrings and tests

**Implementation Notes**:
- Leverage `registry.get_compatible_estimators()` for candidate selection
- Use scoring functions from Phase 4
- Use `NuisanceTuner` from Phase 3
- Follow REFACTORING_PLAN.md Section 7 for API examples
- **CRITICAL**: Docstrings are mandatory before marking phase complete

---

### Phase 6: Refactor InteractiveLinearRegression (Week 6) ⚠️ **TODO**
**Goal**: Implement protocol, integrate with CausalDataset

37. ⚠️ Add `capabilities` property to `InteractiveLinearRegression`
38. ⚠️ Adapt `fit()` to accept `CausalDataset` (current signature expects DataFrame)
39. ⚠️ Add `effect()` method (or adapt existing `predict(mode='cate')`)
40. ⚠️ Ensure `get_params()`/`set_params()` are sklearn-compatible
41. ⚠️ Consider implementing `InferenceProvider` protocol (analytic inference already exists)
42. ⚠️ **Docstrings**: Complete NumPy-style docstrings for all public methods with runnable examples
43. ⚠️ **Tests**: Ensure existing tests pass + new protocol compliance tests

**Deliverable**: Protocol-compliant `InteractiveLinearRegression` with complete docstrings

**Implementation Notes**:
- Existing file at `/home/jadmin/projects/caml/caml/estimators/benchmark/interactive_ols.py`
- Already has formula-based design matrix creation with Patsy
- Already has `_estimate_ate()`, `_estimate_gate()`, `_estimate_cate()` methods
- Already inherits from `BaseCamlEstimator` and `OLSMixin`
- See CODE_EXAMPLES.md Section 3 for detailed current/target signatures
- **DO NOT** break existing functionality - add protocol compliance alongside
- **CRITICAL**: Docstrings are mandatory before marking phase complete

---

### Phase 7: Testing & Documentation (Week 7) 🔶 **TODO**
**Goal**: Comprehensive validation

44. 🔶 Integration tests: Full AutoCATE workflow
45. 🔶 Benchmarking: New vs old AutoCATE performance
46. 🔶 Write migration guide
47. 🔶 Update API documentation
48. 🔶 Create example notebooks
49. 🔶 **Final docstring audit**: Ensure all public APIs have complete NumPy-style docstrings

**Deliverable**: Production-ready v1 with complete documentation

---

## Implementation Progress Summary

| Phase | Status | Completion | Key Deliverables |
|-------|--------|------------|------------------|
| 1. Foundation | ✅ Complete | 100% | CausalDataset, Protocols, Schema |
| 2. Estimator Wrappers | 🔶 TODO | ~7% | 14 EconML wrappers + registry |
| 3. Nuisance Models | 🔶 TODO | 0% | NuisanceTuner, NuisanceSpec |
| 4. Cross-Fitting & Scoring | 🔶 TODO | 0% | CrossFitter, RLoss, DRLoss, Qini |
| 5. AutoCATE | 🔶 TODO | 0% | New AutoCATE with Optuna |
| 6. InteractiveOLS | ⚠️ Partial | 70% | Protocol adaptation needed |
| 7. Documentation | 🔶 TODO | 0% | Tests, docs, examples |
| **TOTAL** | **~30%** | **~30%** | **5-6 weeks remaining** |

---

## 7. API Examples

### 7.1 Basic Usage (New API)

```python
from caml.data.dataset import CausalDataset
from caml.data.data_schema import TreatmentType, OutcomeType
from caml.automl.auto_cate import AutoCATE

# Load your data
import pandas as pd
df = pd.read_csv("data.csv")

# Create CausalDataset
data = CausalDataset.from_dataframe(
    df,
    X=["age", "income", "education"],
    T="treatment",
    Y="outcome",
    W=["region", "year"],  # Controls, not in CATE model
    treatment_type=TreatmentType.BINARY,
    outcome_type=OutcomeType.CONTINUOUS
)

# Validate data
data.validate()

# Create AutoCATE instance
auto_cate = AutoCATE(
    nuisance_time_budget=300,  # 5 min for nuisance tuning (FLAML)
    n_trials=100,              # 100 Optuna trials for CATE selection
    scoring="r_loss",          # Primary metric
    cv=3,
    random_state=42
)

# Fit
auto_cate.fit(
    data,
    estimator_families=["dml", "dr", "meta"]  # Which families to try
)

# Best estimator
print(auto_cate.best_estimator_)
print(auto_cate.best_score_)

# Predict CATE
cate = auto_cate.predict_cate(df[["age", "income", "education"]])

# Predict with confidence intervals
cate, lower, upper = auto_cate.predict_cate(
    df[["age", "income", "education"]],
    return_interval=True,
    alpha=0.05
)

# Estimate ATE
ate_result = auto_cate.estimate_ate(data, return_inference=True)
print(f"ATE: {ate_result.value} ± {ate_result.stderr}")
```

### 7.2 Custom Scoring

```python
from caml.automl.auto_cate import AutoCATE

# Multi-metric scoring
auto_cate = AutoCATE(
    scoring=["r_loss", "qini"],  # Optimize for both
    scoring_weights=[0.7, 0.3],  # Weighted combination
    n_trials=100
)

auto_cate.fit(data)
```

### 7.3 Using NuisanceTuner Standalone

```python
from caml.nuisance.tuner import NuisanceTuner, NuisanceSpec

# Create spec
spec = NuisanceSpec(
    fit_propensity=True,
    fit_outcome=True,
    propensity_config={"time_budget": 200}
)

# Tune nuisances
tuner = NuisanceTuner(time_budget=300, use_ray=True)
tuner.fit(data, spec)

# Access models
propensity_model = tuner.propensity_model_
outcome_model = tuner.outcome_model_

# Use in custom workflow
from caml.scoring.r_loss import RLoss
scorer = RLoss(propensity_model, outcome_model)
score = scorer(my_estimator, data)
```

### 7.4 Direct Estimator Usage

```python
from caml.estimators.wrappers.dml import WrappedLinearDML

# Use estimator directly
estimator = WrappedLinearDML(
    model_y="auto",
    model_t="auto",
    cv=3,
    random_state=42
)

# Check capabilities
print(estimator.capabilities)

# Fit
estimator.fit(data)

# Predict
cate = estimator.predict_cate(data.X)
```

---

## 8. Success Metrics

### 8.1 Correctness
- [ ] All scoring functions validated against synthetic data with known ground truth
- [ ] R-loss, DR-loss produce expected rankings on benchmark datasets
- [ ] Cross-fitting produces unbiased estimates (verified via simulation)
- [ ] Wrapped estimators produce identical outputs to EconML (within numerical precision)

### 8.2 Performance
- [ ] New AutoCATE completes in ≤ 120% time of old AutoCATE
- [ ] Scoring functions are vectorized (no Python loops over observations)
- [ ] Cross-fitting is parallelized where possible
- [ ] Memory usage ≤ old AutoCATE

### 8.3 Code Quality
- [ ] 80%+ test coverage on all new modules
- [ ] Type hints on all public APIs
- [ ] **Docstrings (numpy style) on all public functions - MANDATORY for phase completion**
- [ ] All docstrings include runnable examples using `SyntheticDataGenerator`
- [ ] Pre-commit hooks pass (ruff, mypy)

### 8.4 Usability
- [ ] Migration guide with side-by-side comparisons
- [ ] Example notebooks for common workflows
- [ ] Clear error messages for invalid data
- [ ] Automatic capability filtering (users never see incompatible estimators)

---

## Appendix A: Implementation Notes

### Key Differences from Original Plan

Based on actual implementation inspection (Jan 24, 2026):

1. **Directory Naming**:
   - `sampling/` used instead of `validation/`
   - `registry/` used instead of `modeling/`

2. **File Naming**:
   - `uplift_.py` (with underscore) instead of `uplift.py`
   - `_validation.py` (private module) instead of `validation.py`

3. **Protocol Method Names**:
   - Use `effect()` instead of `predict_cate()` as per `CATEEstimator` protocol
   - This matches EconML's naming convention

4. **Typo in Implementation**:
   - `EstimatorCapabilities` (missing 'i') instead of `EstimatorCapabilities`
   - Consider fixing in future iteration

5. **Bonus Module**:
   - `extensions/` module added (not in original plan)
   - Contains `SyntheticDataGenerator` - very useful for testing!
   - Contains `plots.py` for visualization

6. **Schema Enhancements**:
   - `Estimand` enum added to `data/data_schema.py`
   - `InferenceType` enum in separate `inference/inference_schema.py`

### Testing Strategy

With `SyntheticDataGenerator` available:

```python
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.data.dataset import CausalDataset

# Generate test data
generator = SyntheticDataGenerator(
    n_cont_outcomes=1,
    n_binary_outcomes=0,
    n_cont_modifiers=3,
    n_binary_modifiers=2,
    seed=42
)
df = generator.df

# Create CausalDataset
data = CausalDataset.from_dataframe(
    df,
    X=[c for c in df.columns if "X" in c],
    T="T1_binary",
    Y="Y1_cont",
    W=[c for c in df.columns if "W" in c]
)

# Use for testing scorers, estimators, etc.
```

### Code Style Observations

From implemented files:

1. **Type hints**: Modern syntax used (`list[str]`, `dict`, `tuple[float, float]`)
2. **Dataclasses**: Preferred for data containers (`@dataclass`)
3. **Protocols**: Using `@runtime_checkable` for duck typing
4. **Enums**: Used for categorical types
5. **Validation**: Robust handling of both pandas and numpy arrays
6. **Error messages**: Descriptive with expected vs actual values

## Appendix B: Key Design Decisions

### B.1 Why Build Scoring Instead of Wrapping EconML?
1. **Full control**: Custom implementations allow us to optimize, extend, and debug without upstream dependencies
2. **Flexibility**: Can add new metrics (stability, sensitivity) without waiting for EconML
3. **Consistency**: All scoring follows same patterns, easier to understand and maintain
4. **Performance**: Can optimize for our specific use cases (e.g., vectorized operations)
5. **Independence**: Not tied to EconML's validation API changes

**Trade-offs**:
- More initial implementation work
- Need to validate correctness ourselves
- Must keep up with latest research

**Mitigation**: Comprehensive testing against known ground truth, validation on semi-synthetic data

### B.2 Why Optuna Over FLAML for CATE Selection?

**FLAML strengths**:
- Fast, proven for supervised learning
- Good for nuisance function tuning (classification/regression with standard metrics)

**Optuna strengths**:
- Flexible custom objectives (R-loss, DR-loss require custom computation)
- Conditional search spaces (e.g., only suggest featurizer if estimator supports it)
- Better pruning for expensive objectives
- Easier to implement multi-objective optimization

**Decision**: Use FLAML for nuisance (standard supervised learning), Optuna for CATE (custom objectives)

### B.3 Why CausalDataset Over DataFrame?

**Benefits**:
1. **Type safety**: Know treatment/outcome types at runtime
2. **Validation**: Centralized checks for overlap, missing data, etc.
3. **Metadata**: Store feature names, treatment labels for interpretability
4. **Consistency**: All estimators work with same data contract
5. **Extensibility**: Easy to add clustering, weights, time indices later

**Trade-offs**:
- Users must convert DataFrame → CausalDataset
- One extra step in workflow

**Mitigation**: Simple `from_dataframe()` helper, clear error messages

---

## Next Steps (Updated Jan 24, 2026)

### Immediate Priorities (Phase 2)

1. **Implement DML Wrappers** (`estimators/wrappers/dml.py`)
   - Wrap LinearDML, SparseLinearDML, CausalForestDML, KernelDML
   - Each must implement `CATEEstimator` protocol
   - See REFACTORING_PLAN.md Section 3.3 for wrapper pattern

2. **Implement DR Wrappers** (`estimators/wrappers/dr.py`)
   - Wrap DRLearner, ForestDRLearner
   - Follow same pattern as DML wrappers
   - Test with SyntheticDataGenerator

3. **Implement Meta-Learner Wrappers** (`estimators/wrappers/meta.py`)
   - Wrap SLearner, TLearner, XLearner
   - Simpler than DML/DR (no nuisance models)
   - Test with binary treatment data

4. **Implement ORF Wrappers** (`estimators/wrappers/orf.py`)
   - Wrap DMLOrthoForest, DROrthoForest
   - Test with various treatment types

5. **Implement Estimator Registry** (`registry/registry.py`)
   - Auto-discovery of wrapped estimators
   - Compatibility filtering based on data
   - See CODE_EXAMPLES.md Section 9 for implementation

### Medium-Term (Phases 3-4)

- Extract NuisanceTuner from existing AutoCATE codebase
- Implement cross-fitting and scoring infrastructure
- Build AutoCATE with Optuna backend

### Long-Term (Phases 5-7)

- Refactor AutoCATE with Optuna backend
- Adapt InteractiveLinearRegression to protocols
- Comprehensive testing and documentation

**Estimated completion**: Early March 2026 (5-6 weeks from Jan 24)

---

**Document Control**
- **Author**: OpenCode + User
- **Last Updated**: January 24, 2026
- **Version**: 1.1 (Updated with implementation status)
