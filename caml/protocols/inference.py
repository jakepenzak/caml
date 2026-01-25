"""Inference provider protocol for uncertainty quantification.

Defines ``InferenceProvider`` protocol for CATE estimators that provide statistical
inference (confidence intervals, standard errors). Separate from ``AutoCateEstimator`` to
support flexible composition via native implementation or wrapper-based inference.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from caml.inference import InferenceResult, InferenceType


@runtime_checkable
class InferenceProvider(Protocol):
    """Protocol for estimators providing statistical inference for CATE estimates.

    Defines interface for uncertainty quantification via confidence intervals and
    standard errors. Separate from ``AutoCateEstimator`` to enable flexible composition.

    See Also
    --------
    [`AutoCateEstimator`](estimator.qmd#caml.protocols.estimator.AutoCateEstimator) : Core protocol for CATE estimation.

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
    import numpy as np
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.inference import InferenceType, InferenceResult
    from caml.protocols import AutoCateEstimator, EstimatorCapabilities, InferenceProvider

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

        def effect_inference(self, X, **effect_inference_kwargs):
            cate = self._estimator.effect_inference(X, )
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
