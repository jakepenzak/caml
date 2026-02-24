"""Wrappers for EconML's Meta-learners (S-Learner, T-Learner, X-Learner).

Wraps `SLearner`, `TLearner`, and `XLearner` to implement CaML's
`AutoCateEstimator` and `InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.linear_model import LinearRegression

from caml.automl import (
    NuisanceModelSpec,
    SearchSpace,
    StandardMLSpec,
)
from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.inference import InferenceType
from caml.registry import auto_register

from ..base_estimator import EstimatorCapabilities
from .base_wrapper import BaseEconMLWrapperMixin


@auto_register(name="SLearner", family="meta")
class WrappedSLearner(BaseEconMLWrapperMixin):
    """Wrapper for EconML's S-Learner estimator.

    S-Learner (Single Learner) estimates CATE using a single model that predicts the
    outcome given features and treatment. CATE is estimated by comparing predictions
    under different treatment values. Simple and fast but assumes treatment effect
    heterogeneity can be captured by feature-treatment interactions. Supports binary
    and multi-valued treatments with continuous or binary outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.metalearners.SLearner``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space: SearchSpace
        Default hyperparameter search space for AutoML tuning.

    See Also
    --------
    [EconML SLearner](https://www.pywhy.org/econml/_autosummary/econml.metalearners.SLearner.html) : Official documentation for EconML's SLearner.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import GradientBoostingRegressor

    from caml.estimators.meta import WrappedSLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.utilities.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

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

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=False,
        supports_weights=False,
        requires_treatment_model=False,
        requires_outcome_model=False,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (StandardMLSpec(name="overall_model"),)

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        if "overall_model" not in self._econml_kwargs:
            self._econml_kwargs["overall_model"] = LinearRegression()
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
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            **fit_kwargs,
        )

        self._is_fitted = True

        return self


@auto_register(name="TLearner", family="meta")
class WrappedTLearner(BaseEconMLWrapperMixin):
    """Wrapper for EconML's T-Learner estimator.

    T-Learner (Two Learner) estimates CATE by training separate models for each
    treatment group and taking the difference in predictions. More flexible than
    S-Learner as it allows different outcome models per treatment. Supports binary
    and multi-valued treatments with continuous or binary outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.metalearners.TLearner``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space: SearchSpace
        Default hyperparameter search space for AutoML tuning.

    See Also
    --------
    [EconML TLearner](https://www.pywhy.org/econml/_autosummary/econml.metalearners.TLearner.html) : Official documentation for EconML's TLearner.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import RandomForestRegressor

    from caml.estimators.meta import WrappedTLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.utilities.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

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

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=False,
        supports_weights=False,
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (StandardMLSpec(name="models"),)

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        if "models" not in self._econml_kwargs:
            self._econml_kwargs["models"] = LinearRegression()
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
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
        )

        self._is_fitted = True

        return self


@auto_register(name="XLearner", family="meta")
class WrappedXLearner(BaseEconMLWrapperMixin):
    """Wrapper for EconML's X-Learner estimator.

    X-Learner estimates CATE using a more sophisticated two-stage approach than T-Learner.
    First stage fits outcome models per treatment, second stage fits models for imputed
    treatment effects, which are then combined using propensity score weighting. Generally
    more efficient than T-Learner, especially with imbalanced treatment groups. Supports
    binary treatment only with continuous outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.metalearners.XLearner``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space: SearchSpace
        Default hyperparameter search space for AutoML tuning.

    See Also
    --------
    [EconML XLearner](https://www.pywhy.org/econml/_autosummary/econml.metalearners.XLearner.html) : Official documentation for EconML's XLearner.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegressionCV

    from caml.estimators.meta import WrappedXLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.utilities.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

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

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.CATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=False,
        supports_weights=False,
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=True,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        StandardMLSpec(name="models"),
        StandardMLSpec(name="cate_models"),
        NuisanceModelSpec(name="propensity_model", model_type="treatment"),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        if "models" not in self._econml_kwargs:
            self._econml_kwargs["models"] = LinearRegression()
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
        models. W (confounders) are not supported separately - include them in X if needed.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
        )

        self._is_fitted = True

        return self
