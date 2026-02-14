"""Shared base functionality, protocols, and interfaces for AutoCATE estimators.

Defines the fundamental interfaces that all AutoCATE estimators in CaML must implement.
The protocol-based design enables flexible estimator composition, automatic compatibility
checking, and seamless integration with AutoML workflows.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from caml.automl.search_space import SearchSpace
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

        capabilities.is_compatible(data)
        ```
        """
        return (
            data.treatment_type in self.treatment_types
            and data.outcome_type in self.outcome_types
        )


@runtime_checkable
class AutoCateEstimator(Protocol):
    """Protocol defining the core CATE estimator interface (structural subtyping).

    Specifies the minimal interface all CATE estimators must implement, compatible
    with scikit-learn conventions. Runtime-checkable via ``isinstance(obj, AutoCateEstimator)``.

    Notes
    -----
    This is a Protocol using structural typing - any class implementing these methods
    will satisfy this interface. For a base implementation with validation, parameter
    handling, and helper utilities, see ``BaseAutoCateEstimatorMixin``.

    The ``capabilities`` and ``default_search_space`` attributes must be class attributes, not instance attributes.

    See Also
    --------
    [`BaseAutoCateEstimatorMixin`](base_estimator.qmd#caml.estimators.base_estimator.BaseAutoCateEstimatorMixin) : ABC base class with concrete implementations.

    [`EstimatorCapabilities`](base_estimator.qmd#caml.estimators.base_estimator.EstimatorCapabilities) : Metadata for estimator features.

    [`SearchSpace`](search_space.qmd#caml.automl.search_space.SearchSpace) : Hyperparameter search space for AutoML tuning.
    """

    capabilities: EstimatorCapabilities

    default_search_space: SearchSpace

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check estimator-data compatibility (class method)."""
        ...

    def fit(self, data: CausalDataset, **kwargs):
        """Fit the estimator on causal data."""
        ...

    def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict CATE for given features."""
        ...

    def get_params(self, deep: bool = True) -> dict:
        """Get estimator parameters (scikit-learn compatible)."""
        ...

    def set_params(self, **params):
        """Set estimator parameters (scikit-learn compatible)."""
        ...


@runtime_checkable
class InferenceProvider(Protocol):
    """Protocol for estimators providing statistical inference for CATE estimates.

    Defines interface for uncertainty quantification via confidence intervals and
    standard errors. Separate from ``AutoCateEstimator`` to enable flexible composition.

    Notes
    -----
    This is a structural typing Protocol. Estimators can implement both
    ``AutoCateEstimator`` and ``InferenceProvider`` to provide complete
    CATE estimation with uncertainty quantification.

    For estimators without native inference, use ``BootstrapInferenceWrapper``
    from the samplers module.

    The ``inference_type`` parameter with value ``None`` or ``'auto'`` should
    delegate to the estimator's preferred inference method.
    """

    def effect_inference(
        self,
        X: np.ndarray | pd.DataFrame,
        inference_type: InferenceType | None = None,
        bootstrapper: bool | None = None,
        **effect_inference_kwargs,
    ) -> InferenceResult:
        """Get complete inference results for CATE estimates."""
        ...


class BaseAutoCateEstimatorMixin(ABC, BaseEstimator):
    """Abstract base class for ``AutoCateEstimator`` with validation and utilities.

    Provides concrete implementations of compatibility checking, parameter methods,
    and fitting validation, with utilities inhereted from scikit-learn's `BaseEstimator`
    (e.g., `get_params` and `set_params`).

    Subclasses must implement ``fit()`` and ``effect()`` and define the class attributes ``capabilities`` and ``default_search_space``.

    This class serves as the recommended base for all CATE estimators in CaML,
    providing a consistent interface and common utilities.

    Notes
    -----
    **Abstract Methods (must be implemented by subclasses):**

    - ``fit()`` - Fit the estimator on causal data
    - ``effect()`` - Predict CATE for features

    **Concrete Methods (provided by this base class):**

    - ``is_compatible_with()`` - Class method for compatibility checking
    - ``check_fitted()`` - Internal validation utility

    **Required Class Attributes:**

    - ``capabilities`` - ``EstimatorCapabilities`` instance defining supported features
    - ``default_search_space`` - ``SearchSpace`` instance for AutoML tuning

    Subclasses must define ``capabilities`` and ``default_search_space`` as class attributes. Failure to do so
    will raise a ``TypeError`` on class definition (enforced by ``__init_subclass__``).

    See Also
    --------
    [`AutoCateEstimator`](base_estimator.qmd#caml.estimators.base_estimator.AutoCateEstimator) : Protocol defining the interface.

    [`EstimatorCapabilities`](base_estimator.qmd#caml.estimators.base_estimator.EstimatorCapabilities) : Metadata for estimator features.

    [`SearchSpace`](search_space.qmd#caml.automl.search_space.SearchSpace) : Hyperparameter search space for AutoML tuning.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.estimators import EstimatorCapabilities, BaseAutoCateEstimatorMixin
    from caml.automl import IntSpec

    class SimpleEstimator(BaseAutoCateEstimatorMixin):
        # Required class attribute
        capabilities = EstimatorCapabilities(
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

        default_search_space = (
            IntSpec(name="some_param", low=1, high=100, default=42)
        )

        def __init__(self, some_param: int = 42):
            self.some_param = some_param
            self._is_fitted = False
            self.effect_value = None

        def fit(self, data, **kwargs):
            # Abstract method implementation
            T = np.asarray(data.T)
            Y = np.asarray(data.Y)
            self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
            self._is_fitted = True
            return self

        def effect(self, X, **kwargs):
            # Abstract method implementation
            self.check_fitted()
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.effect_value)

    # Verify protocol conformance
    from caml.estimators import AutoCateEstimator
    est = SimpleEstimator()
    assert isinstance(est, AutoCateEstimator)
    assert isinstance(est, BaseAutoCateEstimatorMixin)

    print(est.set_params(some_param=100))
    print(est.get_params())
    ```

    ```{python}
    # Use is_compatible_with before instantiation
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )

    # Check compatibility before instantiating
    if SimpleEstimator.is_compatible_with(data):
        est = SimpleEstimator()
        est.fit(data)
        cate = est.effect(data.X)
        print(f"CATE estimate: {cate[0]:.3f}")
    ```
    """

    _is_fitted: bool = False

    # Class attribute that must be overridden by subclasses
    capabilities: EstimatorCapabilities
    default_search_space: SearchSpace

    @abstractmethod
    def fit(self, data: CausalDataset, **kwargs) -> AutoCateEstimator:
        """Fit the CATE estimator on causal data (**ABSTRACT**).

        Notes
        -----
        **This is an abstract method.** Subclasses must provide a complete
        implementation that fits the estimator to the provided dataset.

        Implementations should:

        1. Validate data compatibility (optional - can use ``is_compatible_with``)
        2. Fit any nuisance models required (treatment, outcome, regression)
        3. Fit the final CATE model
        4. Set ``self._is_fitted = True``
        5. Return ``self`` for method chaining

        Parameters
        ----------
        data
            Causal dataset with treatment (T), outcome (Y), and covariates (X, W).
        **kwargs
            Estimator-specific arguments (e.g., nuisance models, hyperparameters).

        Returns
        -------
        AutoCateEstimator
            Fitted estimator instance (self).

        Raises
        ------
        ValueError
            If dataset is incompatible with estimator capabilities.

        Examples
        --------
        ```python
        # Typical implementation pattern
        from caml.data import CausalDataset
        from caml.estimators import BaseAutoCateEstimatorMixin


        class MyEstimator(BaseAutoCateEstimatorMixin):
            # ... capabilities definition ...

            def fit(self, data: CausalDataset, **kwargs):
                # 1. Validate compatibility
                if not self.is_compatible_with(data):
                    raise ValueError("Incompatible data")

                # 2. Fit nuisance models (if needed)
                self._fit_treatment_model(data)

                # 3. Fit CATE model
                # ... fit logic ...

                # 4. Mark as fitted
                self._is_fitted = True

                # 5. Return self for chaining
                return self
        ```
        """
        ...

    @abstractmethod
    def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict CATE for given features (**ABSTRACT**).

        Notes
        -----
        **This is an abstract method.** Subclasses must provide a complete
        implementation that predicts CATE for new observations.

        Implementations should:

        1. Call ``self.check_fitted()`` to validate estimator state
        2. Convert input to appropriate format (if needed)
        3. Compute CATE predictions
        4. Return predictions as NumPy array

        Parameters
        ----------
        X
            Feature matrix for CATE prediction. Can be NumPy array or pandas DataFrame.
        **kwargs
            Additional prediction arguments (estimator-specific).

        Returns
        -------
        np.ndarray
            CATE estimates. Shape ``(n_samples,)`` for binary treatment,
            ``(n_samples, n_treatments)`` for multi-valued treatment.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.

        Examples
        --------
        ```python
        # Typical implementation pattern
        import numpy as np
        from caml.estimators import BaseAutoCateEstimatorMixin


        class MyEstimator(BaseAutoCateEstimatorMixin):
            # ... capabilities definition ...

            def effect(self, X, **kwargs):
                # 1. Check if fitted
                self.check_fitted()

                # 2. Convert input format if needed
                X = np.asarray(X)

                # 3. Compute predictions
                # ... prediction logic ...
                cate = np.random.randn(len(X))  # Placeholder for actual predictions

                # 4. Return as array
                return np.asarray(cate)
        ```
        """
        ...

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check estimator-data compatibility (**CONCRETE**).

        This is a class method enabling compatibility checking without instantiation.
        Useful for filtering candidate estimators in AutoML workflows before fitting.

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

        # Generate test data
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
        # Filter multiple estimators efficiently
        from caml.estimators.wrappers.dml import (
            WrappedLinearDML,
            WrappedSparseLinearDML,
        )

        candidates = [WrappedLinearDML, WrappedSparseLinearDML]
        compatible = [
            est_class for est_class in candidates
            if est_class.is_compatible_with(data)
        ]
        print(f"Compatible: {[c.__name__ for c in compatible]}")
        ```
        """
        return cls.capabilities.is_compatible(data)

    def check_fitted(self):
        """Check if estimator has been fitted (**CONCRETE**).

        Raises
        ------
        RuntimeError
            If estimator has no underlying estimator or has not been fitted.

        Notes
        -----
        This is an internal utility method. Subclasses should call this at the
        start of ``effect()`` and other post-fit methods to ensure the estimator
        has been properly fitted.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction. "
                "Call .fit() first."
            )

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce that subclasses define required class attributes (**CONCRETE**).

        Raises
        ------
        TypeError
            If non-abstract subclass doesn't define ``capabilities``.
        """
        super().__init_subclass__(**kwargs)

        if "capabilities" not in cls.__dict__ and not inspect.isabstract(cls):
            raise TypeError(
                f"{cls.__name__} must define 'capabilities' as a class attribute. "
                f"See EstimatorCapabilities for details."
            )
        if "default_search_space" not in cls.__dict__ and not inspect.isabstract(cls):
            raise TypeError(
                f"{cls.__name__} must define 'default_search_space' as a class attribute. "
                f"See SearchSpace for details."
            )

    def _validate_search_space(self):
        for attr in self.default_search_space:
            if attr.name not in self.get_params(deep=False):
                raise TypeError(
                    f"{self} must have an __init__ parameter for "
                    f"'{attr.name}' defined in default_search_space."
                )
