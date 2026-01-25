"""Inference provider protocol for uncertainty quantification.

Defines ``InferenceProvider`` protocol for CATE estimators that provide statistical
inference (confidence intervals, standard errors). Separate from ``CATEEstimator`` to
support flexible composition via native implementation or wrapper-based inference.
"""

from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from caml.inference import InferenceResult


@runtime_checkable
class InferenceProvider(Protocol):
    """Protocol for estimators providing statistical inference for CATE estimates.

    Defines interface for uncertainty quantification via confidence intervals and
    standard errors. Separate from ``CATEEstimator`` to enable flexible composition.

    See Also
    --------
    [`CATEEstimator`](estimator.qmd#caml.protocols.estimator.CATEEstimator) : Core protocol for CATE estimation.

    [`InferenceResult`](results.qmd#caml.inference.results.InferenceResult) : Dataclass for inference outputs.

    [`InferenceType`](inference_schema.qmd#caml.inference.inference_schema.InferenceType) : Enum defining inference method types.

    Notes
    -----
    - Runtime-checkable via ``isinstance(obj, InferenceProvider)``
    - Estimators can implement both ``CATEEstimator`` and ``InferenceProvider``
    - For estimators without native inference, use ``BootstrapInferenceWrapper``
    - Method parameter ``'auto'`` delegates to estimator's preferred inference method

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.inference import InferenceType, InferenceResult
    from caml.protocols import CATEEstimator, EstimatorCapabilities, InferenceProvider

    class InferenceCapableEstimator:
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.ANALYTIC},
            estimands={Estimand.CATE},
            supports_inference=True
        )

        def __init__(self):
            self.effect_value = None
            self.se_value = 0.1

        def fit(self, data, **kwargs):
            T = np.asarray(data.T)
            Y = np.asarray(data.Y)
            self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
            return self

        def effect(self, X, **kwargs):
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.effect_value)

        def effect_interval(self, X, alpha=0.05, method='auto', **kwargs):
            cate = self.effect(X)
            se = np.full_like(cate, self.se_value)
            z = 1.96
            return cate - z * se, cate + z * se

        def effect_stderr(self, X, method='auto', **kwargs):
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.se_value)

        def effect_inference(self, X, alpha=0.05, method='auto', **kwargs):
            cate = self.effect(X)
            se = self.effect_stderr(X, method=method)
            ci_lower, ci_upper = self.effect_interval(X, alpha, method)
            return InferenceResult(
                point_estimate=cate,
                stderr=se,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                alpha=alpha,
                method=InferenceType.ANALYTIC
            )

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    print(isinstance(InferenceCapableEstimator(), InferenceProvider))  # True
    ```
    """

    def effect_interval(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float = 0.05,
        method: Literal["auto", "analytic", "bootstrap"] = "auto",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals for CATE estimates.

        Parameters
        ----------
        X
            Feature matrix for inference.
        alpha
            Significance level (e.g., 0.05 for 95% CI).
        method
            Inference method: ``'auto'``, ``'analytic'``, or ``'bootstrap'``.
        **kwargs
            Additional arguments (e.g., ``n_bootstrap`` for bootstrap).

        Returns
        -------
        ci_lower
            Lower confidence bounds.
        ci_upper
            Upper confidence bounds.

        Raises
        ------
        ValueError
            If method not supported (check ``capabilities.inference_types``).
        """
        ...

    def effect_stderr(
        self,
        X: np.ndarray | pd.DataFrame,
        method: Literal["auto", "analytic", "bootstrap"] = "auto",
        **kwargs,
    ) -> np.ndarray:
        """Compute standard errors for CATE estimates.

        Parameters
        ----------
        X
            Feature matrix for inference.
        method
            Inference method: ``'auto'``, ``'analytic'``, or ``'bootstrap'``.
        **kwargs
            Additional arguments (e.g., ``n_bootstrap`` for bootstrap).

        Returns
        -------
        np.ndarray
            Standard errors for each observation.

        Raises
        ------
        ValueError
            If method not supported.
        """
        ...

    def effect_inference(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float = 0.05,
        method: Literal["auto", "analytic", "bootstrap"] = "auto",
        **kwargs,
    ) -> InferenceResult:
        """Get complete inference results for CATE estimates.

        Returns point estimates, standard errors, confidence intervals, and metadata
        in a single ``InferenceResult`` object.

        Parameters
        ----------
        X
            Feature matrix for inference.
        alpha
            Significance level for confidence intervals.
        method
            Inference method: ``'auto'``, ``'analytic'``, or ``'bootstrap'``.
        **kwargs
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
