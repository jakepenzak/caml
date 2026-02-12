"""Shared base functionality (mixin) for wrapped AutoCATE estimators."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd

from caml.data import CausalDataset
from caml.inference import InferenceResult, InferenceType

from ..base_estimator import AutoCateEstimator, BaseAutoCateEstimatorMixin


class BaseEconMLWrapperMixin(BaseAutoCateEstimatorMixin):
    """Mixin providing common functionality for EconML wrappers."""

    @abstractmethod
    def fit(self, data: CausalDataset, **kwargs) -> AutoCateEstimator: ...

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
