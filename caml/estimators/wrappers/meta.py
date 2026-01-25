"""Wrappers for EconML's Meta-learners (S-Learner, T-Learner, X-Learner).

Wraps `SLearner`, `TLearner`, and `XLearner` to implement CaML's
`AutoCateEstimator` and `InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.metalearners import SLearner, TLearner, XLearner

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators.base import BaseWrapperMixin
from caml.inference import InferenceType
from caml.protocols import EstimatorCapabilities


class WrappedSLearner(BaseWrapperMixin):
    """Wrapper for EconML's S-Learner estimator.

    S-Learner (Single Learner) estimates CATE using a single model that predicts the
    outcome given features and treatment. CATE is estimated by comparing predictions
    under different treatment values. Simple and fast but assumes treatment effect
    heterogeneity can be captured by feature-treatment interactions. Supports binary
    and multi-valued treatments with continuous or binary outcomes.

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.metalearners.SLearner``.
        Common parameters include:

        - overall_model : estimator, optional
            Model for E[Y|X,T] (default: auto-selected).
            The model should support fit(X, y) and predict(X) interface.
        - random_state : int, optional
            Random seed for reproducibility.

    Attributes
    ----------
    capabilities : EstimatorCapabilites
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    clean_name : str
        Human-readable name for the estimator ("SLearner").

    See Also
    --------
    [EconML SLearner](https://www.pywhy.org/econml/_autosummary/econml.metalearners.SLearner.html) : Official documentation for EconML's SLearner.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](estimator.qmd#caml.protocols.estimator.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import GradientBoostingRegressor

    from caml.estimators.wrappers.meta import WrappedSLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.protocols import AutoCateEstimator

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=3,
        n_obs=500,
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

    # Fit S-Learner
    estimator = WrappedSLearner(
        overall_model=GradientBoostingRegressor(n_estimators=100),
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")
    ```
    """

    capabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
        },
        supports_confounders_in_first_stage_only=False,
        supports_weights=False,
        requires_propensity=False,
        supports_inference=True,
    )
    clean_name = "SLearner"

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = SLearner(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedSLearner:
        """Fit the estimator on causal data.

        Parameters
        ----------
        data
            Causal dataset containing X, T, Y.
        **fit_kwargs
            Additional keyword arguments passed to EconML's fit method.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        S-Learner trains a single model on the combined feature space [X, T].
        W (confounders) are not supported separately - include them in X if needed.
        """
        self.check_compatibility(data, raise_error=True)

        init_kwargs = self._econml_kwargs.copy()
        self._estimator = SLearner(**init_kwargs)

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
        )

        self._is_fitted = True

        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedTLearner(BaseWrapperMixin):
    """Wrapper for EconML's T-Learner estimator.

    T-Learner (Two Learner) estimates CATE by training separate models for each
    treatment group and taking the difference in predictions. More flexible than
    S-Learner as it allows different outcome models per treatment. Supports binary
    and multi-valued treatments with continuous or binary outcomes.

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.metalearners.TLearner``.
        Common parameters include:

        - models : estimator or list of estimators, optional
            Model(s) for E[Y|X,T=t] for each treatment value.
            If a single estimator is provided, it's cloned for each treatment.
            If a list, should have one estimator per treatment value.
        - random_state : int, optional
            Random seed for reproducibility.

    Attributes
    ----------
    capabilities : EstimatorCapabilites
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    clean_name : str
        Human-readable name for the estimator ("TLearner").

    See Also
    --------
    [EconML TLearner](https://www.pywhy.org/econml/_autosummary/econml.metalearners.TLearner.html) : Official documentation for EconML's TLearner.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](estimator.qmd#caml.protocols.estimator.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import RandomForestRegressor

    from caml.estimators.wrappers.meta import WrappedTLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.protocols import AutoCateEstimator

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=4,
        n_obs=600,
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

    # Fit T-Learner
    estimator = WrappedTLearner(
        models=RandomForestRegressor(n_estimators=100, max_depth=10),
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")
    ```
    """

    capabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
        },
        supports_confounders_in_first_stage_only=False,
        supports_weights=False,
        requires_propensity=False,
        supports_inference=True,
    )
    clean_name = "TLearner"

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = TLearner(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedTLearner:
        """Fit the estimator on causal data.

        Parameters
        ----------
        data
            Causal dataset containing X, T, Y.
        **fit_kwargs
            Additional keyword arguments passed to EconML's fit method.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        T-Learner trains separate models for each treatment group.
        W (confounders) are not supported separately - include them in X if needed.
        """
        self.check_compatibility(data, raise_error=True)

        init_kwargs = self._econml_kwargs.copy()
        self._estimator = TLearner(**init_kwargs)

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
        )

        self._is_fitted = True

        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedXLearner(BaseWrapperMixin):
    """Wrapper for EconML's X-Learner estimator.

    X-Learner estimates CATE using a more sophisticated two-stage approach than T-Learner.
    First stage fits outcome models per treatment, second stage fits models for imputed
    treatment effects, which are then combined using propensity score weighting. Generally
    more efficient than T-Learner, especially with imbalanced treatment groups. Supports
    binary treatment only with continuous outcomes.

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.metalearners.XLearner``.
        Common parameters include:

        - models : estimator or list of 2 estimators, optional
            Models for E[Y|X,T=t]. If single estimator, cloned for both groups.
        - cate_models : estimator or list of 2 estimators, optional
            Models for imputed treatment effects in second stage.
        - propensity_model : estimator, optional
            Model for propensity score E[T|X] (default: LogisticRegressionCV).
        - random_state : int, optional
            Random seed for reproducibility.

    Attributes
    ----------
    capabilities : EstimatorCapabilites
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    clean_name : str
        Human-readable name for the estimator ("XLearner").

    See Also
    --------
    [EconML XLearner](https://www.pywhy.org/econml/_autosummary/econml.metalearners.XLearner.html) : Official documentation for EconML's XLearner.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](estimator.qmd#caml.protocols.estimator.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegressionCV

    from caml.estimators.wrappers.meta import WrappedXLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.protocols import AutoCateEstimator

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=4,
        n_obs=700,
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

    # Fit X-Learner
    estimator = WrappedXLearner(
        models=GradientBoostingRegressor(n_estimators=100),
        cate_models=GradientBoostingRegressor(n_estimators=50),
        propensity_model=LogisticRegressionCV(),
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")
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
        supports_confounders_in_first_stage_only=False,
        supports_weights=False,
        requires_propensity=True,
        supports_inference=True,
    )
    clean_name = "XLearner"

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = XLearner(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedXLearner:
        """Fit the estimator on causal data.

        Parameters
        ----------
        data
            Causal dataset containing X, T, Y.
        **fit_kwargs
            Additional keyword arguments passed to EconML's fit method.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        X-Learner uses propensity score weighting to combine first and second stage
        models. Only supports binary treatment. W (confounders) are not supported
        separately - include them in X if needed.
        """
        self.check_compatibility(data, raise_error=True)

        init_kwargs = self._econml_kwargs.copy()
        self._estimator = XLearner(**init_kwargs)

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
        )

        self._is_fitted = True

        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
