# CaML AutoCATE Refactoring Plan

**Version:** 1.3
**Date:** January 31, 2026
**Status:** In Progress - Phases 1-4 Complete, Phase 5 Next

---

## Executive Summary

**UPDATE (Jan 31, 2026)**: Phases 1-4 are complete! Core v0 functionality is ready. Some Phase 4 components (bootstrap inference, uplift/policy/calibration scorers) have been deferred to post-v0 as they are not critical for the initial release.

This plan refactors CaML into a focused **AutoCATE modeling package** with:

- **EconML-first approach**: Wrap 14 proven CATE estimators from EconML ✅ COMPLETE
- **Custom scoring infrastructure**: R-loss, DR-loss, Q-statistic, PEHE ✅ COMPLETE (uplift/policy/calibration deferred)
- **Dual AutoML backends**: FLAML for nuisance function tuning ✅ COMPLETE, Optuna for CATE model selection
- **Extracted nuisance tuner**: Reusable component for first-stage model optimization ✅ COMPLETE
- **Cross-fitting engine**: Out-of-fold nuisance predictions for unbiased scoring ✅ COMPLETE
- **Minimal custom estimators**: Only `InteractiveLinearRegression` (benchmark) for v1
- **All treatment types supported**: Binary, multi-valued, continuous from day one ✅ COMPLETE
- **First-class inference**: Confidence intervals and standard errors as core functionality ✅ COMPLETE
- **Protocol-based architecture**: `BaseWrapperMixin` ABC for consistent wrapper patterns ✅ COMPLETE

**Progress**: ~70% complete (Phases 1-4 of 7)
**Timeline**: 2-3 weeks remaining for v0 release
**Lines of Code**: ~2,200 implemented, ~1,000 remaining

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

| **Metric** | **Implementation** | **Status** |
|-----------|-------------------|------------|
| **R-loss** | Custom orthogonal score with cross-fitted nuisances | ✅ **COMPLETE** |
| **DR-loss** | Custom doubly-robust pseudo-outcome loss | ✅ **COMPLETE** |
| **Q-statistic** | IPW pseudo-outcome for model ranking | ✅ **COMPLETE** |
| **PEHE** | Oracle metric with true CATEs | ✅ **COMPLETE** |
| **Qini / AUUC** | Uplift metrics for binary treatment | 🔶 **Deferred to post-v0** |
| **Policy value (DR)** | Doubly-robust policy evaluation | 🔶 **Deferred to post-v0** |
| **Calibration** | CATE calibration by deciles | 🔶 **Deferred to post-v0** |
| **Diagnostics** | Stability, sensitivity, overlap checks | 🔶 **Deferred to post-v0** |

**Implemented Scorers** (in `scorers/`):
- `BaseCateScorerMixin` - Abstract base with validation utilities
- `RLoss` - R-learner loss for model selection
- `DRLoss` - Doubly-robust loss for model selection
- `QStat` - Q-statistic for model ranking
- `PEHE` - Oracle metric (requires true CATEs)

**Deferred Scorers** (prefixed with `_`, empty TODO files):
- `_uplift.py` - Qini, AUUC
- `_policy.py` - Policy value scoring
- `_calibration.py` - CATE calibration
- `_diagnostics.py` - Stability metrics
- `_plug_in.py` - Plug-in estimator

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
│   ├── data_enums.py            # ✅ TreatmentType, OutcomeType, Estimand enums
│   └── _validation.py            # ✅ Overlap, positivity, missing data checks
│
├── protocols/                    # ⚠️ DEPRECATED - Consolidated into estimators/base.py
│   └── (legacy - use estimators/base.py instead)
│
├── estimators/                   # ✅ IMPLEMENTED - CATE estimators
│   ├── __init__.py               # ✅ Exports protocols and wrappers
│   ├── base_estimator.py                   # ✅ NEW - AutoCateEstimator, InferenceProvider, BaseWrapperMixin (697 lines)
│   ├── native/                   # ⚠️ PARTIAL (renamed from benchmark/)
│   │   ├── __init__.py
│   │   └── interactive_ols.py    # ⚠️ EXISTS (24,781 bytes) - Needs protocol adaptation
│   └── wrappers/                 # ✅ IMPLEMENTED - EconML wrappers (Phase 2 complete!)
│       ├── __init__.py           # ✅ Exports all wrappers
│       ├── dml.py                # ✅ IMPLEMENTED (779 lines) - 5 DML wrappers with full docstrings
│       ├── dr.py                 # ✅ IMPLEMENTED (652 lines) - 4 DR wrappers with full docstrings
│       ├── meta.py               # ✅ IMPLEMENTED (439 lines) - 3 meta-learner wrappers with full docstrings
│       └── orf.py                # ✅ IMPLEMENTED (288 lines) - 2 ORF wrappers with full docstrings
│
├── nuisance/                     # ✅ IMPLEMENTED - First-stage nuisance estimation
│   ├── __init__.py               # ✅ IMPLEMENTED (5 lines)
│   ├── tuner.py                  # ✅ IMPLEMENTED (224 lines) - NuisanceTuner (FLAML-based)
│   └── spec.py                   # ✅ IMPLEMENTED (54 lines) - NuisanceTunerSpec dataclass
│
├── scorers/                      # ✅ IMPLEMENTED - Scoring & evaluation (core scorers complete)
│   ├── __init__.py               # ✅ Exports BaseCateScorerMixin, RLoss, DRLoss, QStat, PEHE
│   ├── base_scorer.py            # ✅ BaseCateScorerMixin ABC, _clip(), validation utilities (220 lines)
│   ├── r_loss.py                 # ✅ RLoss scorer (144 lines)
│   ├── dr_loss.py                # ✅ DRLoss scorer (143 lines)
│   ├── q_stat.py                 # ✅ QStat scorer (128 lines)
│   ├── pehe.py                   # ✅ PEHE oracle metric (127 lines)
│   ├── _uplift.py                # 🔶 Deferred - Qini, AUUC (TODO placeholder)
│   ├── _policy.py                # 🔶 Deferred - Policy value (TODO placeholder)
│   ├── _calibration.py           # 🔶 Deferred - CATE calibration (TODO placeholder)
│   ├── _diagnostics.py           # 🔶 Deferred - Stability metrics (TODO placeholder)
│   └── _plug_in.py               # 🔶 Deferred - Plug-in estimator (TODO placeholder)
│
├── samplers/                     # ✅ IMPLEMENTED - Cross-fitting & resampling (core complete)
│   ├── __init__.py               # ✅ Exports CrossFitter, create_splitter
│   ├── cross_fit.py              # ✅ CrossFitter class (297 lines)
│   ├── splitters.py              # ✅ create_splitter utility (40 lines)
│   └── bootstrap.py              # 🔶 Deferred - Bootstrap inference (TODO placeholder)
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
├── inference/                    # ✅ IMPLEMENTED - Inference utilities
│   ├── __init__.py               # ✅ Implemented
│   ├── results.py                # ✅ InferenceResult dataclass
│   └── inference_enums.py       # ✅ InferenceType enum
│
├── registry/                     # ✅ IMPLEMENTED - Model registry (renamed from modeling/)
│   ├── __init__.py               # ✅ IMPLEMENTED (11 lines)
│   ├── model_bank.py             # ✅ IMPLEMENTED (72 lines) - Default estimator bank
│   └── registry.py               # ✅ IMPLEMENTED (163 lines) - Auto-discovery & filtering
│
├── extensions/                   # ✅ IMPLEMENTED - New utilities (not in original plan)
│   ├── __init__.py               # ✅ Implemented
│   ├── synthetic_data.py         # ✅ SyntheticDataGenerator (~48KB)
│   └── plots.py                  # ✅ Plotting utilities
│
├── _generics/                    # ✅ EXISTING - Utilities (keep)
│   ├── logging.py
│   ├── decorators.py
│   ├── monkey_patch.py
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
from caml.data.data_enums import TreatmentType, OutcomeType

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

### 3.2 Estimator Protocols (estimators/base_estimator.py)

**Purpose**: Define interfaces all estimators must satisfy

**Note**: Protocols consolidated into `estimators/base_estimator.py` instead of separate `protocols/` module for better cohesion.

```python
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from caml.data.dataset import CausalDataset
from caml.data.data_enums import TreatmentType, OutcomeType, Estimand
from caml.inference import InferenceType, InferenceResult

@dataclass(frozen=True)
class EstimatorCapabilities:
    """Describes what an estimator supports."""
    treatment_types: set[TreatmentType]
    outcome_types: set[OutcomeType]
    inference_types: set[InferenceType]
    estimands: set[Estimand]
    supports_controls_in_first_stage_only: bool
    supports_weights: bool
    requires_treatment_model: bool
    requires_outcome_model: bool
    requires_regression_model: bool
    supports_inference: bool

    def is_compatible(self, data: CausalDataset) -> bool:
        """Check if estimator can handle the given dataset."""
        return (
            data.treatment_type in self.treatment_types
            and data.outcome_type in self.outcome_types
        )

@runtime_checkable
class AutoCateEstimator(Protocol):
    """Core protocol for CATE estimators."""

    capabilities: EstimatorCapabilities



    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check compatibility without instantiation (for filtering)."""
        ...

    def check_compatibility(self, data: CausalDataset, raise_error: bool = True) -> bool:
        """Check compatibility with detailed error messages."""
        ...

    def fit(self, data: CausalDataset, **kwargs) -> "AutoCateEstimator":
        """Fit the estimator."""
        ...

    def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict CATE for given features."""
        ...

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters (sklearn compatibility)."""
        ...

    def set_params(self, **params) -> "AutoCateEstimator":
        """Set parameters (sklearn compatibility)."""
        ...

@runtime_checkable
class InferenceProvider(Protocol):
    """Protocol for estimators providing statistical inference."""

    def effect_inference(
        self,
        X: np.ndarray | pd.DataFrame,
        inference_type: InferenceType | None = None,
        bootstrapper: bool | None = None,
        **effect_inference_kwargs,
    ) -> InferenceResult:
        """Get complete inference results for CATE estimates."""
        ...
```

### 3.2.1 BaseWrapperMixin (estimators/base_estimator.py)

**Purpose**: Abstract base class providing common wrapper functionality for all EconML wrappers

**Key Innovation**: This mixin wasn't in the original plan but emerged as a critical pattern for consistent wrapper implementation.

```python
from abc import ABC, abstractmethod
from econml._cate_estimator import BaseCateEstimator

class BaseWrapperMixin(ABC):
    """Mixin and ABC providing common functionality for EconML wrappers."""

    _estimator: BaseCateEstimator
    _is_fitted: bool = False

    # Abstract methods (must implement in each wrapper)


    @property
    @abstractmethod
    def capabilities(self) -> EstimatorCapabilities:
        """Estimator capabilities metadata."""
        pass

    @abstractmethod
    def fit(self, data: CausalDataset, **fit_kwargs) -> "BaseWrapperMixin":
        """Fit the estimator on causal data."""
        pass

    @abstractmethod
    def get_params(self, deep: bool = True) -> dict:
        """Get estimator parameters."""
        pass

    @abstractmethod
    def set_params(self, **params) -> dict:
        """Set estimator parameters."""
        pass

    # Concrete methods (inherited by all wrappers)
    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check compatibility without instantiation (for filtering)."""
        temp_instance = cls()
        return temp_instance.capabilities.is_compatible(data)

    def check_compatibility(self, data: CausalDataset, raise_error: bool = True) -> bool:
        """Check compatibility with detailed error messages."""
        is_compatible = self.capabilities.is_compatible(data)

        if not is_compatible and raise_error:
            raise ValueError(
                f"Data incompatible with {self.__class__.__name__}.\n"
                f"  Required treatment types: {self.capabilities.treatment_types}\n"
                f"  Required outcome types: {self.capabilities.outcome_types}\n"
                f"  Got treatment type: {data.treatment_type}\n"
                f"  Got outcome type: {data.outcome_type}"
            )

        return is_compatible

    def effect(self, X: np.ndarray | pd.DataFrame, **effect_kwargs) -> np.ndarray:
        """Predict CATE by delegating to underlying estimator."""
        self._check_fitted()
        return self._estimator.effect(X, **effect_kwargs)

    def effect_inference(
        self,
        X: np.ndarray | pd.DataFrame,
        inference_type: InferenceType | None = None,
        bootstrapper: bool | None = None,
        **effect_inference_kwargs,
    ) -> InferenceResult:
        """Get complete inference results."""
        if inference_type == InferenceType.BOOTSTRAP:
            raise NotImplementedError("Bootstrap inference not yet implemented.")

        effect_inference = self._estimator.effect_inference(X, **effect_inference_kwargs)

        return InferenceResult(
            effect=effect_inference.point_estimate,
            stderr=effect_inference.stderr,
            method=inference_type,
        )

    def __getattr__(self, name: str):
        """Forward attribute access to underlying estimator if not found on wrapper.

        This enables seamless access to EconML-specific methods and attributes.
        For example: wrapper.model_y → wrapper._estimator.model_y
        """
        if self._estimator is not None and hasattr(self._estimator, name):
            return getattr(self._estimator, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def _check_fitted(self):
        """Verify estimator has been fitted before prediction."""
        if self._estimator is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no underlying estimator set."
            )
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction. "
                "Call .fit() first."
            )
```

**Benefits of BaseWrapperMixin**:
1. **DRY principle**: Common logic written once, inherited by 14 wrappers
2. **Attribute delegation**: Seamless access to EconML-specific methods via `__getattr__`
3. **Consistent error handling**: Uniform compatibility checks and fit verification
4. **Dual compatibility checking**: Both class-level (for filtering) and instance-level (for detailed errors)
5. **Type safety**: Abstract methods enforce complete implementation

### 3.3 Estimator Wrapper Pattern (estimators/wrappers/dml.py)

**Purpose**: Wrap EconML estimators to satisfy AutoCateEstimator protocol

**Status**: ✅ IMPLEMENTED - All 14 wrappers follow this pattern

```python
from econml.dml import LinearDML
from caml.estimators.base_estimator import BaseWrapperMixin, EstimatorCapabilities
from caml.data.data_enums import TreatmentType, OutcomeType, Estimand
from caml.data.dataset import CausalDataset
from caml.inference import InferenceType
import numpy as np

class WrappedLinearDML(BaseWrapperMixin):
    """Wrapper for EconML's LinearDML estimator.

    Inherits all common functionality from BaseWrapperMixin including:
    - effect() method (delegates to _estimator.effect())
    - effect_inference() method (wraps to InferenceResult)
    - __getattr__() for attribute delegation
    - is_compatible_with() class method
    - check_compatibility() instance method
    - _check_fitted() validation
    """

    def __init__(self, **econml_kwargs):
        """Initialize with EconML parameters."""
        self._econml_kwargs = econml_kwargs
        self._estimator = LinearDML(**self._econml_kwargs)
        self._is_fitted = False

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """Define what this estimator supports."""
        return EstimatorCapabilities(
            treatment_types={
                TreatmentType.BINARY,
                TreatmentType.CONTINUOUS,
                TreatmentType.MULTI,
            },
            outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
            inference_types={InferenceType.ANALYTIC, InferenceType.BOOTSTRAP},
            estimands={
                Estimand.ATE,
                Estimand.ATT,
                Estimand.ATC,
                Estimand.CATE,
                Estimand.GATE,
            },
            supports_controls_in_first_stage_only=True,
            supports_weights=True,
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            supports_inference=True,
        )


    def fit(self, data: CausalDataset, **fit_kwargs) -> "WrappedLinearDML":
        """Fit using CausalDataset.

        Automatically sets discrete flags based on data metadata.
        """
        # Check compatibility first
        self.check_compatibility(data, raise_error=True)

        # Set discrete flags based on data metadata (auto-configuration)
        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )
        self._estimator.discrete_treatment = (
            True if data.treatment_type.is_discrete() else False
        )

        # Fit underlying EconML estimator
        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
            sample_weight=data.weights if data.weights is not None else None,
            **fit_kwargs,
        )

        self._is_fitted = True
        return self

    def get_params(self, deep=True) -> dict:
        """Get estimator parameters (sklearn-compatible)."""
        return {}  # Simple version - may enhance later

    def set_params(self, **params) -> "WrappedLinearDML":
        """Set estimator parameters (sklearn-compatible)."""
        return self  # Simple version - may enhance later
```

**Key Features**:
1. **Inherits from BaseWrapperMixin**: Gets effect(), effect_inference(), __getattr__(), etc. for free
2. **Auto-configuration**: Sets discrete_treatment/discrete_outcome flags from CausalDataset metadata
3. **Comprehensive capabilities**: Explicitly declares all supported types via EstimatorCapabilities
4. **Clean error messages**: check_compatibility() provides detailed mismatch information
5. **Attribute delegation**: Can access wrapper.model_y, wrapper.const_marginal_effect(), etc. via __getattr__
6. **NumPy docstrings**: All public methods documented with runnable examples

### 3.4 Nuisance Tuner (nuisance/tuner.py)

**Status**: ✅ **IMPLEMENTED** (224 lines)

**Purpose**: Extract nuisance model tuning from AutoCATE

```python
from dataclasses import dataclass
from flaml import AutoML
from caml.data.dataset import CausalDataset
import pandas as pd

@dataclass
class NuisanceTunerSpec:
    """Specification for which nuisance models to fit."""
    fit_treatment_model: bool | None = None  # E[T|X,W] - propensity score
    fit_outcome_model: bool | None = None     # E[Y|X,W]
    fit_regression_model: bool | None = None  # E[Y|X,W,T] for DR methods

    # FLAML config overrides
    treatment_model_config: dict | None = None
    outcome_model_config: dict | None = None
    regression_model_config: dict | None = None

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
        self.treatment_model_ = None
        self.outcome_model_ = None
        self.regression_model_ = None

    def fit(self, data: CausalDataset, spec: NuisanceTunerSpec) -> "NuisanceTuner":
        """Fit nuisance models based on spec."""

        base_config = self._build_base_config()

        if spec.fit_treatment_model:
            config = self._build_treatment_model_config(data, base_config, spec)
            self.treatment_model_ = self._run_flaml(config)

        if spec.fit_outcome_model:
            config = self._build_outcome_model_config(data, base_config, spec)
            self.outcome_model_ = self._run_flaml(config)

        if spec.fit_regression_model:
            config = self._build_regression_model_config(data, base_config, spec)
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

    def _build_treatment_model_config(self, data: CausalDataset, base: dict, spec: NuisanceTunerSpec) -> dict:
        """Build config for treatment model E[T|X,W] (propensity score model)."""
        config = base.copy()

        # Determine task type
        if data.treatment_type.is_discrete():
            config["task"] = "classification"
            config["metric"] = "log_loss"
        else:
            config["task"] = "regression"
            config["metric"] = "mse"

        # Prepare data (concatenate X and W)
        XW = pd.concat([data.X, data.W], axis=1) if data.W is not None else data.X
        config["X_train"] = XW
        config["y_train"] = data.T

        # Apply user overrides
        if spec.treatment_model_config:
            config.update(spec.treatment_model_config)

        return config

    def _build_outcome_model_config(self, data: CausalDataset, base: dict, spec: NuisanceTunerSpec) -> dict:
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
        if spec.outcome_model_config:
            config.update(spec.outcome_model_config)

        return config

    def _build_regression_model_config(self, data: CausalDataset, base: dict, spec: NuisanceTunerSpec) -> dict:
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
        if spec.regression_model_config:
            config.update(spec.regression_model_config)

        return config

    def _run_flaml(self, config: dict):
        """Run AutoML and return best estimator."""
        automl = AutoML()
        automl.fit(**config)
        return automl.model.estimator
```

**Implementation Notes**:
- Uses `treatment_model_`, `outcome_model_`, `regression_model_` (not `propensity_model_`)
- Automatically detects task type (classification vs regression) from `CausalDataset` metadata
- Supports Ray/Spark distributed tuning via FLAML
- Config overrides allow per-model FLAML customization
- Feature preparation (concatenating X and W) integrated directly into config builder methods

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
2. ✅ Create `data/data_enums.py` - `TreatmentType`, `OutcomeType`, `Estimand` enums
3. ✅ Create `data/_validation.py` - validation functions
4. ✅ Create `protocols/estimator.py` - `CATEEstimator` protocol, `EstimatorCapabilities`
5. ✅ Create `inference/results.py` - `InferenceResult` dataclass
6. ✅ Create `inference/inference_enums.py` - `InferenceType` enum
7. ✅ Create `protocols/inference.py` - `InferenceProvider` protocol
8. ✅ **Tests**: Validate `CausalDataset.from_dataframe()`, validation logic

**Deliverable**: ✅ Working `CausalDataset` with comprehensive tests

**Notes**:
- Implementation uses `EstimatorCapabilities` (typo - missing 'i')
- Protocol uses `effect()` method name instead of `predict_cate()`
- Bonus: `extensions/` module added with `SyntheticDataGenerator` for testing

---

### Phase 2: Estimator Wrappers (Week 2) ✅ **COMPLETE**
**Goal**: Wrap EconML estimators ✅ ACHIEVED

**Implemented**:
1. ✅ `estimators/base_estimator.py` - AutoCateEstimator, InferenceProvider, EstimatorCapabilities, BaseWrapperMixin (697 lines)
2. ✅ `estimators/wrappers/dml.py` - 5 DML wrappers (LinearDML, SparseLinearDML, CausalForestDML, NonParamDML, KernelDML) (779 lines)
3. ✅ `estimators/wrappers/dr.py` - 4 DR wrappers (DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner) (652 lines)
4. ✅ `estimators/wrappers/meta.py` - 3 meta-learners (SLearner, TLearner, XLearner) (439 lines)
5. ✅ `estimators/wrappers/orf.py` - 2 ORF estimators (DMLOrthoForest, DROrthoForest) (288 lines)
6. ✅ `registry/registry.py` - Estimator discovery & filtering (163 lines)
7. ✅ `registry/model_bank.py` - Default estimator bank (72 lines)
8. ✅ **NumPy docstrings**: All wrappers have complete docstrings with runnable examples using SyntheticDataGenerator
9. ✅ **Tests**: Comprehensive test coverage for all wrappers (tests/caml/estimators/wrappers/)

**Key Implementation Details**:

1. **BaseWrapperMixin Pattern**:
   - All 14 wrappers inherit from `BaseWrapperMixin` ABC
   - Provides automatic `__getattr__` delegation to underlying EconML estimator
   - Dual compatibility checking (class method + instance method)
   - Unified `effect()` and `effect_inference()` interfaces
   - Consistent error handling and fit verification

2. **Registry Pattern**:
   - `get_compatible_estimators(data, families=None)` returns dict of compatible estimator classes
   - `register_estimator(name, estimator, family)` adds custom estimators to global registry
   - Automatic filtering by treatment/outcome type compatibility
   - Structure: `{"name": {"estimator": Class, "family": "dml"}}`

3. **Auto-Configuration**:
   - Wrappers automatically set `discrete_treatment`/`discrete_outcome` flags from CausalDataset metadata
   - No manual configuration needed for treatment/outcome types

5. **Attribute Delegation**:
   - All EconML-specific attributes/methods accessible via wrapper
   - Example: `wrapper.model_y` → `wrapper._estimator.model_y`
   - Example: `wrapper.const_marginal_effect()` → `wrapper._estimator.const_marginal_effect()`

**Deliverable**: ✅ 14 wrapped estimators + registry system (100% complete)

**Notes**:
- Originally planned `protocols/` module consolidated into `estimators/base.py` for better cohesion
- `BaseWrapperMixin` emerged as critical architectural pattern (not in original plan)
- All wrappers follow identical structure for maintainability
- Protocols moved from separate module to `estimators/base.py`

---

### Phase 3: Nuisance Models (Week 3) ✅ **COMPLETE**
**Goal**: Extract nuisance model tuning from AutoCATE ✅ ACHIEVED

**Implemented**:
1. ✅ `nuisance/spec.py` - `NuisanceTunerSpec` dataclass (54 lines)
2. ✅ `nuisance/tuner.py` - `NuisanceTuner` with FLAML (224 lines)
3. ✅ Refactored to use `CausalDataset` with automatic task detection
4. ✅ **Docstrings**: Complete NumPy-style docstrings with runnable examples using `SyntheticDataGenerator`
5. ✅ **Tests**: Comprehensive tests (302 lines for tuner.py, 156 lines for spec.py)

**Deliverable**: ✅ Standalone `NuisanceTuner` with complete docstrings and tests

**Implementation Notes**:
- Uses `treatment_model_`, `outcome_model_`, `regression_model_` (not `propensity_model_`)
- Spec uses `fit_treatment_model`, `fit_outcome_model`, `fit_regression_model` (not `fit_propensity`)
- Automatically detects task type (classification vs regression) from `CausalDataset` metadata
- Supports Ray/Spark distributed tuning via FLAML
- Config overrides allow per-model FLAML customization via `treatment_model_config`, etc.
- Feature preparation (concatenating X and W) integrated directly into tuner methods
- Originally planned `models.py` helper file was not needed - functionality integrated into tuner

---

### Phase 4: Cross-Fitting & Scoring (Week 4) ✅ **COMPLETE** (core scorers)
**Goal**: Custom scoring infrastructure ✅ ACHIEVED

**Implemented**:
1. ✅ `samplers/splitters.py` - Splitting strategies (40 lines)
2. ✅ `samplers/cross_fit.py` - `CrossFitter` class with DML and DR nuisance methods (297 lines)
3. ✅ `scorers/base_scorer.py` - `BaseCateScorerMixin` ABC with validation utilities (220 lines)
4. ✅ `scorers/r_loss.py` - `RLoss` scorer with normalization option (144 lines)
5. ✅ `scorers/dr_loss.py` - `DRLoss` scorer with normalization option (143 lines)
6. ✅ `scorers/q_stat.py` - `QStat` scorer for IPW-based ranking (128 lines)
7. ✅ `scorers/pehe.py` - `PEHE` oracle metric requiring true CATEs (127 lines)
8. ✅ **Docstrings**: Complete NumPy-style docstrings with runnable examples
9. ✅ **Tests**: Validation on synthetic data with `SyntheticDataGenerator`

**Deferred to post-v0**:
- 🔶 `scorers/_uplift.py` - Qini, AUUC metrics
- 🔶 `scorers/_policy.py` - Policy value scoring
- 🔶 `scorers/_calibration.py` - Calibration diagnostics
- 🔶 `scorers/_diagnostics.py` - Stability metrics
- 🔶 `scorers/_plug_in.py` - Plug-in estimator
- 🔶 `samplers/bootstrap.py` - Bootstrap inference

**Deliverable**: ✅ Working scoring module with core metrics (R-loss, DR-loss, Q-stat, PEHE)

**Key Implementation Details**:

1. **CrossFitter Pattern**:
   - Uses `sklearn.model_selection.cross_val_predict` for efficient cross-fitting
   - Supports stratified splitting for discrete outcomes/treatments
   - Methods: `fit_predict_nuisances_dml()`, `fit_predict_nuisances_dr()`, `fit_predict_outcome_model()`, `fit_predict_treatment_model()`, `fit_predict_regression_model()`

2. **Scorer Base Class**:
   - `BaseCateScorerMixin` ABC with `__call__(estimator, data) -> float`
   - Utility functions: `_clip()`, `validate_cate_array()`, `validate_scorer_inputs()`
   - All scorers support `normalized` parameter for R²-like interpretation

3. **Scorer API Consistency**:
   - All scorers take `estimator` (fitted) and `data` (CausalDataset)
   - Use `estimator.effect(data.X)` to get CATE predictions
   - Return float scores (losses, not negated)

---

### Phase 5: Refactor AutoCATE (Week 5) 🔶 **TODO** - **NEXT PRIORITY**
**Goal**: New AutoCATE with Optuna

**Dependencies**: Phases 3-4 complete (nuisance tuner and scorers available)

30. 🔶 Create `automl/backends/base.py` - `TunerBackend` protocol
31. 🔶 Create `automl/backends/optuna_backend.py` - Optuna implementation
32. 🔶 Create `automl/objectives.py` - Optuna objective functions using RLoss/DRLoss
33. 🔶 Create `automl/search_space.py` - Search space definitions
34. 🔶 Refactor `automl/auto_cate.py` - Use `NuisanceTuner` + Optuna + registry + scorers
35. 🔶 **Docstrings**: Complete NumPy-style docstrings for all public classes/methods with runnable examples
36. 🔶 **Tests**: End-to-end AutoCATE on synthetic data

**Deliverable**: New AutoCATE with complete docstrings and tests

**Implementation Notes**:
- Leverage `registry.get_compatible_estimators()` for candidate selection
- Use `NuisanceTuner` from Phase 3 for first-stage models
- Use `RLoss`, `DRLoss`, `QStat` from Phase 4 for scoring
- Use `CrossFitter` for out-of-fold nuisance predictions
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
- Existing file at `/home/jadmin/projects/caml/caml/estimators/native/interactive_ols.py` (note: `native/` not `benchmark/`)
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
| 1. Foundation | ✅ Complete | 100% | CausalDataset, Protocols, Schema, Validation |
| 2. Estimator Wrappers | ✅ Complete | 100% | 14 EconML wrappers + BaseWrapperMixin + registry |
| 3. Nuisance Models | ✅ Complete | 100% | NuisanceTuner, NuisanceTunerSpec |
| 4. Cross-Fitting & Scoring | ✅ Complete | 85% | CrossFitter, RLoss, DRLoss, QStat, PEHE (uplift/policy/calibration deferred) |
| 5. AutoCATE | 🔶 TODO | 0% | New AutoCATE with Optuna - **NEXT PRIORITY** |
| 6. InteractiveOLS | ⚠️ Partial | 70% | Protocol adaptation needed |
| 7. Documentation | 🔶 TODO | 0% | Tests, docs, examples |
| **TOTAL** | **~70%** | **~70%** | **2-3 weeks remaining for v0** |

### Deferred to Post-v0 Release

The following components are not critical for v0 and have been deferred:

| Component | File | Reason |
|-----------|------|--------|
| Bootstrap Inference | `samplers/bootstrap.py` | Analytic inference sufficient for v0 |
| Qini/AUUC Metrics | `scorers/_uplift.py` | R-loss/DR-loss sufficient for model selection |
| Policy Value Scorer | `scorers/_policy.py` | Advanced use case |
| Calibration Metrics | `scorers/_calibration.py` | Advanced diagnostics |
| Stability Diagnostics | `scorers/_diagnostics.py` | Advanced diagnostics |
| Plug-in Estimator | `scorers/_plug_in.py` | Alternative scoring approach |

These files exist as `# TODO` placeholders with underscore prefix to indicate deferred status.

---

## 7. API Examples

### 7.1 Basic Usage (New API)

```python
from caml.data.dataset import CausalDataset
from caml.data.data_enums import TreatmentType, OutcomeType
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
- [x] All Phase 1-2 modules have 80%+ test coverage
- [x] Wrapped estimators produce identical outputs to EconML (within numerical precision)
- [x] BaseWrapperMixin delegation tested across all wrapper types
- [x] Dual compatibility checking (class + instance methods) validated
- [x] All scoring functions validated against synthetic data with known ground truth
- [x] R-loss, DR-loss, Q-stat produce expected rankings on benchmark datasets
- [x] Cross-fitting produces unbiased estimates (verified via simulation)
- [ ] AutoCATE end-to-end tests pass

### 8.2 Performance
- [ ] New AutoCATE completes in ≤ 120% time of old AutoCATE
- [x] Scoring functions are vectorized (no Python loops over observations)
- [x] Cross-fitting uses sklearn cross_val_predict (parallelized)
- [ ] Memory usage ≤ old AutoCATE

### 8.3 Code Quality
- [x] 80%+ test coverage on Phases 1-4 modules
- [x] Type hints on all public APIs
- [x] **Docstrings (numpy style) on all Phase 1-4 public functions**
- [x] All docstrings include runnable examples using `SyntheticDataGenerator`
- [ ] Pre-commit hooks pass (ruff, mypy)

### 8.4 Usability
- [ ] Migration guide with side-by-side comparisons
- [ ] Example notebooks for common workflows
- [x] Clear error messages for invalid data
- [x] Automatic capability filtering (users never see incompatible estimators)

---

## Appendix A: Implementation Notes

### Key Differences from Original Plan

Based on actual implementation inspection (Jan 25, 2026):

1. **Directory Naming**:
   - `samplers/` used instead of `sampling/` or `validation/`
   - `scorers/` used instead of `scoring/`
   - `registry/` used instead of `modeling/`
   - `estimators/native/` used instead of `estimators/benchmark/`

2. **File Naming**:
   - `uplift_.py` (with underscore) instead of `uplift.py`
   - `_validation.py` (private module) instead of `validation.py`

3. **Protocol Consolidation**:
   - Protocols consolidated into `estimators/base_estimator.py` instead of separate `protocols/` module
   - Better cohesion and discoverability
   - Single source of truth for estimator interfaces

4. **Method Names**:
   - Use `effect()` instead of `predict_cate()` as per `AutoCateEstimator` protocol
   - This matches EconML's naming convention

5. **New Architectural Patterns**:
   - `BaseWrapperMixin` ABC (not in original plan) - critical innovation for DRY principle
   - Attribute delegation via `__getattr__` for seamless EconML access
   - Dual compatibility checking (class + instance methods)
   - Auto-configuration of discrete flags from CausalDataset metadata

6. **Bonus Module**:
   - `extensions/` module added (not in original plan)
   - Contains `SyntheticDataGenerator` - very useful for testing!
   - Contains `plots.py` for visualization

7. **Schema Enhancements**:
   - `Estimand` enum added to `data/data_enums.py`
   - `InferenceType` enum in separate `inference/inference_enums.py`
   - More comprehensive `EstimatorCapabilities` with 10 attributes

### Phase 2 Implementation Patterns Observed

All 14 wrappers follow this identical structure:

```python
class WrappedEstimator(BaseWrapperMixin):
    """Wrapper for EconML's Estimator.

    Complete NumPy docstrings with runnable examples.
    """

    def __init__(self, **econml_kwargs):
        """Store kwargs and instantiate EconML estimator."""
        self._econml_kwargs = econml_kwargs
        self._estimator = EconMLEstimator(**econml_kwargs)
        self._is_fitted = False

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """Define comprehensive capabilities."""
        return EstimatorCapabilities(
            treatment_types={...},
            outcome_types={...},
            inference_types={...},
            estimands={...},
            supports_controls_in_first_stage_only=...,
            supports_weights=...,
            requires_treatment_model=...,
            requires_outcome_model=...,
            requires_regression_model=...,
            supports_inference=...,
        )



    def fit(self, data: CausalDataset, **fit_kwargs):
        """Fit with auto-configuration.

        Automatically sets discrete flags based on data metadata.
        """
        # 1. Check compatibility
        self.check_compatibility(data, raise_error=True)

        # 2. Auto-configure discrete flags (if applicable)
        if hasattr(self._estimator, 'discrete_outcome'):
            self._estimator.discrete_outcome = data.outcome_type.is_discrete()
        if hasattr(self._estimator, 'discrete_treatment'):
            self._estimator.discrete_treatment = data.treatment_type.is_discrete()

        # 3. Fit underlying estimator
        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
            sample_weight=data.weights if data.weights is not None else None,
            **fit_kwargs,
        )

        # 4. Mark as fitted
        self._is_fitted = True
        return self

    def get_params(self, deep=True) -> dict:
        """Get parameters (simple version for now)."""
        return {}

    def set_params(self, **params):
        """Set parameters (simple version for now)."""
        return self
```

**Key Pattern Benefits**:
1. **Consistency**: All 14 wrappers follow identical structure
2. **Maintainability**: Changes to BaseWrapperMixin propagate to all wrappers
3. **Auto-configuration**: No manual setting of discrete flags
4. **Clean separation**: Wrapper logic vs. EconML delegation clearly separated
5. **Type safety**: Protocol enforcement via ABC abstract methods

### Attribute Delegation Pattern

All wrappers inherit `__getattr__` from BaseWrapperMixin:

```python
# These all work seamlessly via delegation:
wrapper.model_y                    # → wrapper._estimator.model_y
wrapper.model_t                    # → wrapper._estimator.model_t
wrapper.const_marginal_effect()    # → wrapper._estimator.const_marginal_effect()
wrapper.effect_interval()          # → wrapper._estimator.effect_interval()
wrapper.shap_values()              # → wrapper._estimator.shap_values()
# etc. - ANY EconML attribute/method is accessible
```

This enables seamless access to EconML-specific functionality while maintaining clean wrapper interface.

### Dual Compatibility Checking Pattern

**Class Method** (for filtering without instantiation):
```python
# Check compatibility WITHOUT creating instance
if WrappedLinearDML.is_compatible_with(data):
    estimator = WrappedLinearDML()
    estimator.fit(data)
```

**Instance Method** (for detailed error messages):
```python
# Check compatibility WITH detailed errors
estimator = WrappedLinearDML()
try:
    estimator.check_compatibility(data, raise_error=True)
except ValueError as e:
    print(f"Incompatibility: {e}")
    # Prints:
    # Data incompatible with WrappedLinearDML.
    #   Required treatment types: {TreatmentType.BINARY, TreatmentType.CONTINUOUS, TreatmentType.MULTI}
    #   Required outcome types: {OutcomeType.CONTINUOUS, OutcomeType.BINARY}
    #   Got treatment type: TreatmentType.BINARY
    #   Got outcome type: OutcomeType.BINARY
```

### Registry Structure Pattern

```python
# Registry returns dict with metadata
available_estimators = {
    "LinearDML": {
        "estimator": WrappedLinearDML,  # Class, not instance!
        "family": "dml"
    },
    "DRLearner": {
        "estimator": WrappedDRLearner,
        "family": "dr"
    },
    # ... 12 more
}

# Get compatible estimators
compatible = get_compatible_estimators(data, families=["dml", "dr"])
# Returns: {"LinearDML": {"estimator": WrappedLinearDML, "family": "dml"}, ...}

# Can instantiate later
for name, info in compatible.items():
    est = info["estimator"]()  # Instantiate
    est.fit(data)
```

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

## Next Steps (Updated Jan 31, 2026)

### Immediate Priorities (Phase 5 - AutoML)

**Status Change**: Phases 1-4 are COMPLETE. Phase 5 (AutoCATE with Optuna) is now the immediate priority.

1. **Implement TunerBackend Protocol** (`automl/backends/base.py`)
   - Protocol defining `optimize(objective, n_trials, **kwargs)` interface
   - ~30 lines of code

2. **Implement OptunaBackend** (`automl/backends/optuna_backend.py`)
   - Optuna-based implementation of TunerBackend
   - TPE sampler with pruning
   - ~80 lines of code

3. **Implement Optuna Objectives** (`automl/objectives.py`)
   - Create objectives using RLoss, DRLoss, QStat scorers
   - ~100 lines of code

4. **Implement AutoCATE** (`automl/auto_cate.py`)
   - Main orchestration class
   - Uses NuisanceTuner for first-stage models
   - Uses registry for estimator discovery
   - Uses Optuna for CATE model selection
   - ~300 lines of code

5. **Complete NumPy Docstrings**
   - All public classes and methods need comprehensive docstrings
   - Include runnable examples using `SyntheticDataGenerator`
   - Follow patterns from Phase 4 scorers

6. **Write Tests**
   - End-to-end AutoCATE workflow tests
   - Validate best estimator selection on synthetic data
   - Test multi-metric optimization

### Medium-Term (Phases 6-7)

- Adapt InteractiveLinearRegression to protocols
- Comprehensive testing and documentation
- Example notebooks

### Post-v0 Enhancements

After v0 release, implement deferred components:
- `samplers/bootstrap.py` - Bootstrap inference
- `scorers/_uplift.py` - Qini, AUUC metrics
- `scorers/_policy.py` - Policy value scoring
- `scorers/_calibration.py` - Calibration diagnostics
- `scorers/_diagnostics.py` - Stability metrics

**Estimated v0 completion**: Mid-February 2026 (2-3 weeks from Jan 31)

---

**Document Control**
- **Author**: OpenCode + User
- **Last Updated**: January 31, 2026
- **Version**: 1.3 (Phases 1-4 COMPLETE - Updated with Phase 4 completion and deferred items)
