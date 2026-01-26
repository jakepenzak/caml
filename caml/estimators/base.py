"""Shared base functionality, protocols, and interfaces for CATE estimator wrappers.

Defines the fundamental interfaces that all AutoCATE estimators in CaML must implement.
The protocol-based design enables flexible estimator composition, automatic compatibility
checking, and seamless integration with AutoML workflows.
"""

from __future__ import annotations

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
        If True, confounders (W) used only in nuisance models, not final CATE prediction.
    supports_weights
        If True, estimator handles sample weights.
    requires_treatment_model
        If True, estimator needs a treatment model - $\mathbb{E}[T|X,W]$.
    requires_outcome_model
        If True, estimator needs an outcome model - $\mathbb{E}[Y|X,W]$.
    requires_regression_model
        If True, estimator needs a regression model - $\mathbb{E}[Y|T,X,W]$.
    supports_inference
        If True, estimator implements ``InferenceProvider`` protocol.

    See Also
    --------
    [`TreatmentType`](data_schema.qmd#caml.data.data_schema.TreatmentType) : Treatment variable categories.

    [`OutcomeType`](data_schema.qmd#caml.data.data_schema.OutcomeType) : Outcome variable categories.

    [`InferenceType`](inference_schema.qmd#caml.inference.inference_schema.InferenceType) : Inference method categories.

    [`Estimand`](data_schema.qmd#caml.data.data_schema.Estimand) : Target estimand types.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol using this metadata.

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

        print(capabilities.is_compatible(data))  # True
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

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing what the estimator supports.

    Notes
    -----
    - This is a Protocol (structural subtyping), not a base class
    - Runtime-checkable via ``isinstance(obj, AutoCateEstimator)``
    - Method name is ``effect()`` not ``predict_cate()`` per CaML conventions

    See Also
    --------
    [`EstimatorCapabilities`](base.qmd#caml.estimators.base.EstimatorCapabilities) : Metadata for estimator capabilities.

    [`InferenceProvider`](base.qmd#caml.estimators.base.InferenceProvider) : Additional protocol for inference.

    [`CausalDataset`](dataset.qmd#caml.data.dataset.CausalDataset) : Data container for CATE estimation.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.estimators import AutoCateEstimator, EstimatorCapabilities

    class SimpleEstimator:

        def __init__(self):
            self.effect_value = None

        @property
        def clean_name(self) -> str:
            return "SimpleEstimator"

        @property
        def capabilities(self) -> EstimatorCapabilities:
            return EstimatorCapabilities(
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

        def is_compatible_with(cls, data: CausalDataset) -> bool:
            temp_instance = cls()
            return temp_instance.capabilities.is_compatible(data)

        def check_compatibility(
            self, data: CausalDataset, raise_error: bool = True
        ) -> bool:
            is_compatible = self.capabilities.is_compatible(data)
            return is_compatible

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

    @property
    def clean_name(self) -> str:
        """Human-readable name of the estimator."""
        ...

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """Estimator capabilities metadata."""
        ...

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

    def check_compatibility(
        self, data: CausalDataset, raise_error: bool = True
    ) -> bool:
        """Check compatibility and optionally raise detailed error (instance method).

        Parameters
        ----------
        data
            Dataset to check.
        raise_error
            If True, raise ValueError with detailed message on incompatibility.
            If False, return boolean.

        Returns
        -------
        bool
            True if compatible (only returned if raise_error=False).

        Raises
        ------
        ValueError
            If incompatible and raise_error=True. Error message includes details
            about what's required vs what was provided.
        """
        ...

    def fit(self, data: CausalDataset, **kwargs) -> AutoCateEstimator:
        """Fit the CATE estimator on causal data.

        Parameters
        ----------
        data
            Causal dataset with treatment (T), outcome (Y), and covariates (X, W).
        **kwargs
            Estimator-specific arguments (e.g., nuisance model hyperparameters).

        Returns
        -------
        AutoCateEstimator
            Fitted estimator instance (self).

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

    See Also
    --------
    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Core protocol for CATE estimation.

    [`InferenceResult`](results.qmd#caml.inference.results.InferenceResult) : Dataclass for inference outputs.

    [`InferenceType`](inference_schema.qmd#caml.inference.inference_schema.InferenceType) : Enum defining inference method types.

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


class BaseWrapperMixin(ABC):
    """Mixin and ABC providing common functionality and strict inerface enforcement for EconML wrappers."""

    _estimator: BaseCateEstimator
    _is_fitted: bool = False

    @property
    @abstractmethod
    def clean_name(self) -> str:
        """Human-readable name of the estimator."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> EstimatorCapabilities:
        """Estimator capabilities metadata."""
        pass

    @abstractmethod
    def fit(self, data: CausalDataset, **fit_kwargs) -> BaseWrapperMixin:
        """Fit the estimator on causal data."""
        pass

    @abstractmethod
    def get_params(self, deep: bool = True) -> dict:
        """Get estimator parameters (scikit-learn compatible)."""
        pass

    @abstractmethod
    def set_params(self, **params) -> dict:
        """Set estimator parameters (scikit-learn compatible)."""
        pass

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
        from caml.estimators.wrappers.dml import WrappedLinearDML
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
        temp_instance = cls()
        return temp_instance.capabilities.is_compatible(data)

    def check_compatibility(
        self, data: CausalDataset, raise_error: bool = True
    ) -> bool:
        """Check compatibility and optionally raise detailed error (instance method).

        Parameters
        ----------
        data
            Dataset to check.
        raise_error
            If True, raise ValueError with detailed message on incompatibility.
            If False, return boolean.

        Returns
        -------
        bool
            True if compatible (only returned if raise_error=False).

        Raises
        ------
        ValueError
            If incompatible and raise_error=True. Error message includes details
            about what's required vs what was provided.

        Examples
        --------
        ```{python}
        from caml.estimators.wrappers.dml import WrappedLinearDML
        from caml.data import CausalDataset, TreatmentType, OutcomeType
        import numpy as np

        # Create incompatible data (binary outcome, but LinearDML needs continuous)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.binomial(1, 0.5, 100),  # Binary outcome
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.BINARY  # LinearDML needs CONTINUOUS
        )

        est = WrappedLinearDML()

        # Check without raising (for conditional logic)
        is_ok = est.check_compatibility(data, raise_error=False)
        print(f"Compatible: {is_ok}")

        # Check with raising (for validation in fit())
        try:
            est.check_compatibility(data, raise_error=True)
        except ValueError as e:
            print(f"Error: {e}")
        ```
        """
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
        """Predict CATE for given features.

        Parameters
        ----------
        X
            Feature matrix.
        **effect_kwargs
            Additional arguments passed to EconML's effect().
            For discrete treatment: T0, T1 specify treatment comparison (default 0 vs 1).
            For continuous treatment: T0, T1 specify dose levels to compare.

        Returns
        -------
        np.ndarray
            Estimated CATE.
        """
        self._check_fitted()
        return self._estimator.effect(X, **effect_kwargs)

    def effect_inference(
        self,
        X: np.ndarray | pd.DataFrame,
        inference_type: InferenceType | None = None,
        bootstrapper: bool | None = None,
        **effect_inference_kwargs,
    ) -> InferenceResult:
        """Get complete inference results for CATE estimates.

        Returns results in a single ``InferenceResult`` object, which can be used for hypothesis testing and confidence interval generation.

        **TODO: Implement Bootstrapper & cache functionality**

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
        if inference_type == InferenceType.BOOTSTRAP:
            raise NotImplementedError("Bootstrap inference not yet implemented.")

        effect_inference = self._estimator.effect_inference(
            X, **effect_inference_kwargs
        )

        return InferenceResult(
            effect=effect_inference.point_estimate,
            stderr=effect_inference.stderr,
            method=inference_type,
        )

    def __getattr__(self, name: str):
        """Forward attribute access to underlying estimator if not found on Wrapper."""
        if self._estimator is not None and hasattr(self._estimator, name):
            return getattr(self._estimator, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

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
