"""Core protocols for CATE estimators.

Defines the fundamental interfaces that all CATE estimators in CaML must implement.
The protocol-based design enables flexible estimator composition, automatic compatibility
checking, and seamless integration with AutoML workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.inference import InferenceType


@dataclass(frozen=True)
class EstimatorCapabilities:
    """Metadata describing an estimator's supported features and requirements.

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
    supports_confounders_in_first_stage_only
        If True, confounders (W) used only in nuisance models, not final CATE prediction.
    supports_weights
        If True, estimator handles sample weights.
    requires_propensity
        If True, estimator requires propensity score estimates.
    supports_inference
        If True, estimator implements ``InferenceProvider`` protocol.

    See Also
    --------
    [`TreatmentType`](data_schema.qmd#caml.data.data_schema.TreatmentType) : Treatment variable categories.

    [`OutcomeType`](data_schema.qmd#caml.data.data_schema.OutcomeType) : Outcome variable categories.

    [`InferenceType`](inference_schema.qmd#caml.inference.inference_schema.InferenceType) : Inference method categories.

    [`Estimand`](data_schema.qmd#caml.data.data_schema.Estimand) : Target estimand types.

    [`AutoCateEstimator`](estimator.qmd#caml.protocols.estimator.AutoCateEstimator) : Protocol using this metadata.

    Examples
    --------
    ```{python}
    from caml.data import TreatmentType, OutcomeType, Estimand
    from caml.inference import InferenceType
    from caml.protocols import EstimatorCapabilities

    capabilities = EstimatorCapabilities(
        treatment_types={TreatmentType.BINARY},
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.ANALYTIC},
        estimands={Estimand.CATE},
        supports_inference=True
    )
    ```
    """

    treatment_types: set[TreatmentType]
    outcome_types: set[OutcomeType]
    inference_types: set[InferenceType]
    estimands: set[Estimand]
    supports_confounders_in_first_stage_only: bool = False
    supports_weights: bool = False
    requires_propensity: bool = False
    supports_inference: bool = False

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
        from caml.protocols import EstimatorCapabilities
        import numpy as np

        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE}
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
    [`EstimatorCapabilities`](estimator.qmd#caml.protocols.estimator.EstimatorCapabilities) : Metadata for estimator capabilities.

    [`InferenceProvider`](inference.qmd#caml.protocols.inference.InferenceProvider) : Additional protocol for inference.

    [`CausalDataset`](dataset.qmd#caml.data.dataset.CausalDataset) : Data container for CATE estimation.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.protocols import AutoCateEstimator, EstimatorCapabilities

    class SimpleEstimator:
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE}
        )

        clean_name = "SimpleEstimator"

        def __init__(self):
            self.effect_value = None

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

    print(isinstance(SimpleEstimator(), AutoCateEstimator))  # True
    ```
    """

    capabilities: EstimatorCapabilities
    clean_name: str

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
