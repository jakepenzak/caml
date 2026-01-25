"""Wrappers for EconML's Orthogonal Random Forest estimators.

Wraps `DMLOrthoForest` and `DROrthoForest` to implement CaML's
`AutoCateEstimator` and `InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.orf import DMLOrthoForest, DROrthoForest

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators.base import BaseWrapperMixin
from caml.inference import InferenceType
from caml.protocols import EstimatorCapabilities


class WrappedDMLOrthoForest(BaseWrapperMixin):
    """Wrapper for EconML's DMLOrthoForest estimator.

    DMLOrthoForest estimates CATE using an orthogonal random forest with DML-style
    debiasing. The forest adaptively learns treatment effect heterogeneity while
    maintaining honesty (separate samples for tree structure and leaf estimation).
    Supports binary and continuous treatments with continuous outcomes.

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.orf.DMLOrthoForest``.
        Common parameters include:

        - n_trees : int, optional
            Number of trees in the forest (default: 500).
        - max_depth : int, optional
            Maximum depth of trees (default: None, unlimited).
        - min_leaf_size : int, optional
            Minimum number of samples in a leaf (default: 10).
        - max_splits : int, optional
            Maximum number of splits to consider for each feature (default: 10).
        - subsample_ratio : float, optional
            Ratio of samples to use for each tree (default: 0.7).
        - bootstrap : bool, optional
            Whether to use bootstrap sampling (default: False).
        - lambda_reg : float, optional
            Regularization parameter for ridge regression in leaves (default: 0.01).
        - model_T : estimator, optional
            Model for treatment nuisance E[T|X,W] (default: auto-selected).
        - model_Y : estimator, optional
            Model for outcome nuisance E[Y|X,W] (default: auto-selected).
        - n_jobs : int, optional
            Number of parallel jobs (default: -1).
        - random_state : int, optional
            Random seed for reproducibility.

    Attributes
    ----------
    capabilities : EstimatorCapabilites
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    clean_name : str
        Human-readable name for the estimator ("DMLOrthoForest").

    See Also
    --------
    [EconML DMLOrthoForest](https://www.pywhy.org/econml/_autosummary/econml.orf.DMLOrthoForest.html) : Official documentation for EconML's DMLOrthoForest.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](estimator.qmd#caml.protocols.estimator.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    from caml.estimators.wrappers.orf import WrappedDMLOrthoForest
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.protocols import AutoCateEstimator

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=5,
        n_obs=1000,
        seed=42
    )

    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )

    # Fit orthogonal forest
    estimator = WrappedDMLOrthoForest(
        n_trees=200,
        max_depth=50,
        min_leaf_size=10,
        subsample_ratio=0.7,
        model_T=LogisticRegressionCV(),
        model_Y=LassoCV(),
        random_state=42
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")
    print(f"CATE Range: [{cate.min():.3f}, {cate.max():.3f}]")
    ```
    """

    capabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.CONTINUOUS,
        },
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
        },
        supports_confounders_in_first_stage_only=True,
        supports_weights=False,
        requires_propensity=True,
        supports_inference=True,
    )
    clean_name = "DMLOrthoForest"

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = DMLOrthoForest(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedDMLOrthoForest:
        """Fit the estimator on causal data.

        Parameters
        ----------
        data
            Causal dataset containing X, T, Y, and optionally W.
        **fit_kwargs
            Additional keyword arguments passed to EconML's fit method.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        DMLOrthoForest uses honest splitting: separate samples for tree structure
        learning and leaf value estimation. This provides valid inference without
        overfitting bias.
        """
        self.check_compatibility(data, raise_error=True)

        init_kwargs = self._econml_kwargs.copy()

        init_kwargs["discrete_treatment"] = (
            True if data.treatment_type.is_discrete() else False
        )

        self._estimator = DMLOrthoForest(**init_kwargs)

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
        )

        self._is_fitted = True

        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedDROrthoForest(BaseWrapperMixin):
    """Wrapper for EconML's DROrthoForest estimator.

    DROrthoForest estimates CATE using an orthogonal random forest with doubly robust
    debiasing. Combines the adaptive learning of ORF with the robustness of DR methods.
    Provides robustness to misspecification of either propensity or outcome model.
    Supports binary and continuous treatments with continuous outcomes.

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.orf.DROrthoForest``.
        Common parameters include:

        - n_trees : int, optional
            Number of trees in the forest (default: 500).
        - max_depth : int, optional
            Maximum depth of trees (default: None, unlimited).
        - min_leaf_size : int, optional
            Minimum number of samples in a leaf (default: 10).
        - max_splits : int, optional
            Maximum number of splits to consider for each feature (default: 10).
        - subsample_ratio : float, optional
            Ratio of samples to use for each tree (default: 0.7).
        - bootstrap : bool, optional
            Whether to use bootstrap sampling (default: False).
        - lambda_reg : float, optional
            Regularization parameter for ridge regression in leaves (default: 0.01).
        - propensity_model : estimator, optional
            Model for propensity score E[T|X,W] (default: auto-selected).
        - model_Y : estimator, optional
            Model for outcome E[Y|X,W,T] (default: auto-selected).
        - n_jobs : int, optional
            Number of parallel jobs (default: -1).
        - random_state : int, optional
            Random seed for reproducibility.

    Attributes
    ----------
    capabilities : EstimatorCapabilites
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    clean_name : str
        Human-readable name for the estimator ("DROrthoForest").

    See Also
    --------
    [EconML DROrthoForest](https://www.pywhy.org/econml/_autosummary/econml.orf.DROrthoForest.html) : Official documentation for EconML's DROrthoForest.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](estimator.qmd#caml.protocols.estimator.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    from caml.estimators.wrappers.orf import WrappedDROrthoForest
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.protocols import AutoCateEstimator

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=5,
        n_obs=1000,
        seed=42
    )

    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )

    # Fit doubly robust orthogonal forest
    estimator = WrappedDROrthoForest(
        n_trees=200,
        max_depth=50,
        min_leaf_size=10,
        subsample_ratio=0.7,
        propensity_model=LogisticRegressionCV(),
        model_Y=LassoCV(),
        random_state=42
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")
    print(f"CATE Range: [{cate.min():.3f}, {cate.max():.3f}]")
    ```
    """

    capabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
        },
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
        },
        supports_confounders_in_first_stage_only=True,
        supports_weights=False,
        requires_propensity=True,
        supports_inference=True,
    )
    clean_name = "DROrthoForest"

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = DROrthoForest(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedDROrthoForest:
        """Fit the estimator on causal data.

        Parameters
        ----------
        data
            Causal dataset containing X, T, Y, and optionally W.
        **fit_kwargs
            Additional keyword arguments passed to EconML's fit method.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        DROrthoForest combines honest random forest splitting with doubly robust
        debiasing, providing both adaptive heterogeneity learning and robustness
        to model misspecification.
        """
        self.check_compatibility(data, raise_error=True)

        init_kwargs = self._econml_kwargs.copy()

        self._estimator = DROrthoForest(**init_kwargs)

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
        )

        self._is_fitted = True

        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
