"""Shared base functionality, protocols, and interfaces for AutoCATE estimators.

Defines the fundamental interfaces that all AutoCATE estimators in CaML must implement.
The protocol-based design enables flexible estimator composition, automatic compatibility
checking, and seamless integration with AutoML workflows.
"""
# TODO: Rethink exact protocol, ABC, etc. structure and relationships

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from econml._cate_estimator import BaseCateEstimator

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.inference import InferenceResult, InferenceType


@dataclass(frozen=True)
class EstimatorCapabilities:
    r"""Metadata describing an estimator's supported features and requirements.

    Defines what data types, inference methods, and estimands a CATE estimator supports.
    Enables automatic compatibility checking and estimator filtering in AutoML pipelines.

    Parameters
    ----------
    treatment_types
        Treatment variable types the estimator supports (e.g., ``{TreatmentType.BINARY}``).
    outcome_types
        Outcome variable types the estimator supports (e.g., ``{OutcomeType.CONTINUOUS}``).
    inference_types
        Inference methods the estimator provides (e.g., ``{InferenceType.ANALYTIC}``).
        Empty set if estimator doesn't implement ``InferenceProvider``.
    estimands
        Target causal quantities the estimator estimates (typically ``{Estimand.CATE}``).
    supports_controls_in_first_stage_only
        If True, confounders (W) can be used only in nuisance models, not final CATE model.
    supports_weights
        If True, estimator handles sample weights.
    requires_treatment_model
        If True, estimator needs a treatment model - $\mathbb{E}[T \mid X,W]$.
    requires_outcome_model
        If True, estimator needs an outcome model - $\mathbb{E}[Y \mid X,W]$.
    requires_regression_model
        If True, estimator needs a regression model - $\mathbb{E}[Y \mid T,X,W]$.
    supports_inference
        If True, estimator implements ``InferenceProvider`` protocol.

    Examples
    --------
    ```{python}
    from caml.data import TreatmentType, OutcomeType, Estimand
    from caml.inference import InferenceType
    from caml.estimators import EstimatorCapabilities

    capabilities = EstimatorCapabilities(
        treatment_types={TreatmentType.BINARY},
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.ANALYTIC},
        estimands={Estimand.CATE},
        supports_controls_in_first_stage_only=False,
        supports_weights=True,
        requires_treatment_model=True,
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True
    )
    ```
    """

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

    # TODO: Refine compatibility logic!!
    def is_compatible(self, data: CausalDataset) -> bool:
        """Check if estimator can handle the given dataset.

        Parameters
        ----------
        data
            Dataset to check compatibility with.

        Returns
        -------
        bool
            True if estimator supports the dataset's treatment and outcome types.

        Examples
        --------
        ```{python}
        from caml.data import TreatmentType, OutcomeType, Estimand, CausalDataset
        from caml.estimators import EstimatorCapabilities
        import numpy as np

        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=True,
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            supports_inference=False
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS
        )

        print(capabilities.is_compatible(data))
        ```
        """
        return (
            data.treatment_type in self.treatment_types
            and data.outcome_type in self.outcome_types
        )


@runtime_checkable
class AutoCateEstimator(Protocol):
    """Core protocol defining the interface for CATE estimators.

    All CATE estimators in CaML must implement this protocol. Defines minimal interface
    for fitting and prediction, compatible with scikit-learn conventions.

    Notes
    -----
    - This is a Protocol (structural subtyping), not a base class
    - Runtime-checkable via ``isinstance(obj, AutoCateEstimator)``
    - ``capabilities`` is class attributes, not properties

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.estimators import AutoCateEstimator, EstimatorCapabilities

    class SimpleEstimator:
        # Class attributes
        capabilities: EstimatorCapabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False
        )

        def __init__(self):
            self.effect_value = None

        @classmethod
        def is_compatible_with(cls, data: CausalDataset) -> bool:
            return cls.capabilities.is_compatible(data)

        def fit(self, data, **kwargs):
            T = np.asarray(data.T)
            Y = np.asarray(data.Y)
            self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
            return self

        def effect(self, X, **kwargs):
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.effect_value)

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    assert isinstance(SimpleEstimator(), AutoCateEstimator)
    ```
    """

    capabilities: EstimatorCapabilities

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check if estimator can handle the given dataset (class method).

        This is a class method, so you can check compatibility without
        instantiating the estimator. Useful for filtering candidate
        estimators in AutoML workflows.

        Parameters
        ----------
        data
            Dataset to check compatibility with.

        Returns
        -------
        bool
            True if estimator supports the dataset's treatment and outcome types.
        """
        ...

    def fit(self, data: CausalDataset, **kwargs):
        """Fit the CATE estimator on causal data.

        Parameters
        ----------
        data
            Causal dataset with treatment (T), outcome (Y), and covariates (X, W).
        **kwargs
            Estimator-specific arguments (e.g., nuisance model hyperparameters).

        Raises
        ------
        ValueError
            If dataset is incompatible with estimator capabilities.
        """
        ...

    def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict CATE for given features.

        Parameters
        ----------
        X
            Feature matrix for CATE prediction.
        **kwargs
            Additional prediction arguments.

        Returns
        -------
        np.ndarray
            CATE estimates. Shape (n_samples,) for binary treatment.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.

        Notes
        -----
        For binary treatments: returns E[Y(1) - Y(0) | X].
        """
        ...

    def get_params(self, deep: bool = True) -> dict:
        """Get estimator parameters (scikit-learn compatible).

        Parameters
        ----------
        deep
            If True, return parameters of nested estimators.

        Returns
        -------
        dict
            Parameter names and values.
        """
        ...

    def set_params(self, **params) -> dict:
        """Set estimator parameters (scikit-learn compatible).

        Parameters
        ----------
        **params
            Parameter names and values to set.

        Returns
        -------
        dict
            Estimator instance (self).
        """
        ...


@runtime_checkable
class InferenceProvider(Protocol):
    """Protocol for estimators providing statistical inference for CATE estimates.

    Defines interface for uncertainty quantification via confidence intervals and
    standard errors. Separate from ``AutoCateEstimator`` to enable flexible composition.

    Notes
    -----
    - Runtime-checkable via ``isinstance(obj, InferenceProvider)``
    - Estimators can implement both ``AutoCateEstimator`` and ``InferenceProvider``
    - For estimators without native inference, use ``BootstrapInferenceWrapper``
    - Method parameter ``'auto'`` delegates to estimator's preferred inference method

    Examples
    --------
    ```{python}
    from caml.inference import InferenceType, InferenceResult
    from caml.estimators import InferenceProvider

    class InferenceCapableEstimator:

        def __init__(self):
            self.effect_value = None
            self.se_value = 0.1

        def effect_inference(self, X, **effect_inference_kwargs) -> InferenceResult:
            cate = self._estimator.effect_inference(X)
            return InferenceResult(
                effect=cate,
                stderr=se,
                method=InferenceType.ANALYTIC
            )

    assert isinstance(InferenceCapableEstimator(), InferenceProvider)
    ```
    """

    def effect_inference(
        self,
        X: np.ndarray | pd.DataFrame,
        inference_type: InferenceType | None = None,
        bootstrapper: bool | None = None,
        **effect_inference_kwargs,
    ) -> InferenceResult:
        """Get complete inference results for CATE estimates.

        Returns results in a single ``InferenceResult`` object, which can be used for hypothesis testing and confidence interval generation.

        Parameters
        ----------
        X
            Feature matrix for inference.
        inference_type
            Inference method to use (``InferenceType.ANALYTIC``, ``InferenceType.BOOTSTRAP``, or ``None`` for auto-selection).
        bootstrapper
            Bootstrap sampler to use if ``inference_type`` is ``InferenceType.BOOTSTRAP``. If ``None``, uses default bootstrapper.
        **effect_inference_kwargs
            Additional arguments (e.g., ``n_bootstrap``, ``random_state``).

        Returns
        -------
        InferenceResult
            Complete inference results with point estimates, CIs, stderr, and metadata.

        Raises
        ------
        ValueError
            If method not supported.
        """
        ...


class BaseAutoCateEstimatorMixin(ABC):
    """Base class and mixin for `AutoCateEstimator`."""

    _estimator: BaseCateEstimator
    _is_fitted: bool = False

    # Class attributes that must be overridden by subclasses
    capabilities: EstimatorCapabilities

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check if estimator can handle the given dataset (class method).

        This is a class method, so you can check compatibility without
        instantiating the estimator. Useful for filtering candidate
        estimators in AutoML workflows.

        Parameters
        ----------
        data
            Dataset to check compatibility with.

        Returns
        -------
        bool
            True if estimator supports the dataset's treatment and outcome types.

        Examples
        --------
        ```{python}
        from caml.estimators.dml import WrappedLinearDML
        from caml.data import CausalDataset, TreatmentType, OutcomeType
        from caml.extensions.synthetic_data import SyntheticDataGenerator

        # Generate data
        gen = SyntheticDataGenerator(seed=42)
        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS
        )

        # Check compatibility WITHOUT instantiating
        if WrappedLinearDML.is_compatible_with(data):
            print("LinearDML can handle this data!")
            est = WrappedLinearDML()
            est.fit(data)
        ```

        ```{python}
        # Check multiple estimators efficiently
        from caml.estimators.dml import (
            WrappedLinearDML,
        )

        candidates = [WrappedLinearDML]
        compatible = [
            est_class for est_class in candidates
            if est_class.is_compatible_with(data)
        ]
        print(f"Compatible estimators: {[c.__name__ for c in compatible]}")
        ```
        """
        return cls.capabilities.is_compatible(data)

    @abstractmethod
    def fit(self, data: CausalDataset, **fit_kwargs) -> AutoCateEstimator:
        """Fit the estimator on causal data."""
        ...

    @abstractmethod
    def effect(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict/estimate CATE for given features."""
        ...

    @abstractmethod
    def get_params(self, deep: bool = True) -> dict:
        """Get estimator parameters (scikit-learn compatible)."""
        ...

    @abstractmethod
    def set_params(self, **params) -> dict:
        """Set estimator parameters (scikit-learn compatible)."""
        ...

    def _check_fitted(self):
        """Check if estimator has been fitted."""
        if self._estimator is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no underlying estimator set."
            )
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction. "
                "Call .fit() first."
            )

    def __init_subclass__(cls, **kwargs) -> None:
        """Strictly enforce that subclasses define required class attributes (capabilities)."""
        super().__init_subclass__(**kwargs)

        if "capabilities" not in cls.__dict__ and not inspect.isabstract(cls):
            raise TypeError(f"{cls.__name__} must define capabilities")
