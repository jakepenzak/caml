"""Shared base functionality for CATE estimator wrappers."""

import numpy as np
import pandas as pd
from econml._cate_estimator import BaseCateEstimator

from caml.data import CausalDataset
from caml.inference import InferenceResult, InferenceType
from caml.protocols import EstimatorCapabilities


class BaseWrapperMixin:
    """Mixin providing common functionality for EconML wrappers.

    Provides helper methods for:
    - Checking estimator compatibility with datasets
    - Ensuring estimator is fitted before prediction
    - Forwarding method calls to the underlying estimator, if not present on Wrapper
    """

    capabilities: EstimatorCapabilities
    _is_fitted: bool = False
    _estimator: BaseCateEstimator | None = None

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
