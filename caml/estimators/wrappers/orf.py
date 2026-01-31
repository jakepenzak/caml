"""Wrappers for EconML's Orthogonal Random Forest estimators.

Wraps `DMLOrthoForest` and `DROrthoForest` to implement CaML's
`AutoCateEstimator` and `InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.orf import DMLOrthoForest, DROrthoForest

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators.base import BaseWrapperMixin, EstimatorCapabilities
from caml.inference import InferenceType
from caml.registry import auto_register


@auto_register(family="orf")
class WrappedDMLOrthoForest(BaseWrapperMixin):
    """Wrapper for EconML's DMLOrthoForest estimator.

    DMLOrthoForest estimates CATE using an orthogonal random forest with DML-style
    debiasing. The forest adaptively learns treatment effect heterogeneity while
    maintaining honesty (separate samples for tree structure and leaf estimation).
    Supports binary and continuous treatments with continuous outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.orf.DMLOrthoForest``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    clean_name : str
        Human-readable name for the estimator ("DMLOrthoForest").

    See Also
    --------
    [EconML DMLOrthoForest](https://www.pywhy.org/econml/_autosummary/econml.orf.DMLOrthoForest.html) : Official documentation for EconML's DMLOrthoForest.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    #| echo: false

    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    from caml.estimators.orf import WrappedDMLOrthoForest
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

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
    ```
    """

    # Class attributes
    clean_name: str = "DMLOrthoForest"
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.CONTINUOUS,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=True,
        supports_weights=False,
        requires_treatment_model=True,
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

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

        self._estimator.discrete_treatment = (
            True if data.treatment_type.is_discrete() else False
        )

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


@auto_register(family="orf")
class WrappedDROrthoForest(BaseWrapperMixin):
    """Wrapper for EconML's DROrthoForest estimator.

    DROrthoForest estimates CATE using an orthogonal random forest with doubly robust
    debiasing. Combines the adaptive learning of ORF with the robustness of DR methods.
    Provides robustness to misspecification of either propensity or outcome model.
    Supports binary and continuous treatments with continuous outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.orf.DROrthoForest``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    clean_name : str
        Human-readable name for the estimator ("DROrthoForest").

    See Also
    --------
    [EconML DROrthoForest](https://www.pywhy.org/econml/_autosummary/econml.orf.DROrthoForest.html) : Official documentation for EconML's DROrthoForest.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    #| echo: false
    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    from caml.estimators.orf import WrappedDROrthoForest
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

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
    ```
    """

    # Class attributes
    clean_name: str = "DROrthoForest"
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={TreatmentType.BINARY, TreatmentType.MULTI},
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=True,
        supports_weights=False,
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=True,
        supports_inference=True,
    )

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
