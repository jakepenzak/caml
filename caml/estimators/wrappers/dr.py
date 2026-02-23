"""Wrappers for EconML's Doubly Robust (DR) learners (e.g., DML under the Interactive Regression Model regime).

Wraps `DRLearner`, `LinearDRLearner`, `SparseLinearDRLearner`, and `ForestDRLearner`
to implement CaML's `AutoCateEstimator` and `InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner

from caml.automl import (
    BoolSpec,
    CategoricalSpec,
    ConstantSpec,
    FloatSpec,
    IntSpec,
    NuisanceModelSpec,
    SearchSpace,
    StandardMLSpec,
)
from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.inference import InferenceType
from caml.registry import auto_register

from ..base_estimator import EstimatorCapabilities
from .base_wrapper import BaseEconMLWrapperMixin


@auto_register(name="DRLearner", family="dr")
class WrappedDRLearner(BaseEconMLWrapperMixin):
    """Wrapper for EconML's DRLearner estimator.

    DRLearner estimates CATE using doubly robust learning with flexible model choices
    for propensity, outcome regression, and final CATE model. Provides robustness to
    misspecification of either the propensity or outcome model. Supports binary and
    multi-valued treatments with continuous outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dr.DRLearner``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for AutoML tuning of this estimator.

    See Also
    --------
    [EconML DRLearner](https://www.pywhy.org/EconML/_autosummary/econml.dr.DRLearner.html) : Official documentation for EconML's DRLearner.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    from caml.estimators.dr import WrappedDRLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=4,
        n_obs=800,
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

    # Fit DR learner
    estimator = WrappedDRLearner(
        model_propensity=RandomForestClassifier(n_estimators=50),
        model_regression=RandomForestRegressor(n_estimators=50),
        model_final=RandomForestRegressor(n_estimators=100),
        cv=3,
        random_state=42
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
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.CATE,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=True,
        supports_weights=True,
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=True,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_propensity", model_type="treatment"),
        NuisanceModelSpec(name="model_regression", model_type="regression"),
        StandardMLSpec(name="model_final"),
        FloatSpec(name="min_propensity", lower=1e-6, upper=0.01, log=True),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = DRLearner(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedDRLearner:
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
        DRLearner automatically handles discrete treatments. This method prepares
        the data in the format expected by EconML.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
            sample_weight=data.weights if data.weights is not None else None,
            **fit_kwargs,
        )

        self._is_fitted = True

        return self


@auto_register(name="LinearDRLearner", family="dr")
class WrappedLinearDRLearner(BaseEconMLWrapperMixin):
    """Wrapper for EconML's LinearDRLearner estimator.

    LinearDRLearner estimates CATE using doubly robust learning with a linear final model.
    Provides analytic confidence intervals via debiased moment conditions. More efficient
    than DRLearner when linear CATE assumption is reasonable. Supports binary and multi-valued
    treatments with continuous outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dr.LinearDRLearner``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for AutoML tuning of this estimator.

    See Also
    --------
    [EconML LinearDRLearner](https://www.pywhy.org/econml/_autosummary/econml.dr.LinearDRLearner.html) : Official documentation for EconML's LinearDRLearner.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LogisticRegression, LassoCV

    from caml.estimators.dr import WrappedLinearDRLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=3,
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

    # Fit linear DR learner
    estimator = WrappedLinearDRLearner(
        model_propensity=LogisticRegression(),
        model_regression=LassoCV(),
        cv=3,
        random_state=42
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE with inference
    cate = estimator.effect(data.X)
    effect_inference = estimator.effect_inference(data.X)

    print(f"Mean CATE: {cate.mean():.3f}")
    print(f"Mean StdErr: {effect_inference.stderr.mean():.3f}")
    ```
    """

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.ANALYTIC, InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.CATE,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=True,
        supports_weights=True,
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=True,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_propensity", model_type="treatment"),
        NuisanceModelSpec(name="model_regression", model_type="regression"),
        # CategoricalSpec(
        #     name="featurizer",
        #     choices=[
        #         None,
        #         RobustScaler(),  # Just scaling (no polynomials)
        #         Pipeline(
        #             [
        #                 ("scaler", RobustScaler()),
        #                 ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        #             ]
        #         ),
        #         Pipeline(
        #             [
        #                 ("scaler", RobustScaler()),
        #                 (
        #                     "poly",
        #                     PolynomialFeatures(
        #                         degree=2, interaction_only=True, include_bias=False
        #                     ),
        #                 ),
        #             ]
        #         ),
        #     ],
        # ),
        BoolSpec(name="fit_cate_intercept"),
        FloatSpec(name="min_propensity", lower=1e-6, upper=0.01, log=True),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = LinearDRLearner(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedLinearDRLearner:
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
        LinearDRLearner automatically handles discrete treatments and provides
        analytic inference via debiased moment conditions.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
            sample_weight=data.weights if data.weights is not None else None,
            **fit_kwargs,
        )

        self._is_fitted = True

        return self


@auto_register(name="SparseLinearDRLearner", family="dr")
class WrappedSparseLinearDRLearner(BaseEconMLWrapperMixin):
    """Wrapper for EconML's SparseLinearDRLearner estimator.

    SparseLinearDRLearner estimates CATE using doubly robust learning with a sparse
    linear (Lasso) final model for feature selection. Provides analytic confidence
    intervals via debiased Lasso. Supports binary and multi-valued treatments with
    continuous outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dr.SparseLinearDRLearner``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for AutoML tuning of this estimator.

    See Also
    --------
    [EconML SparseLinearDRLearner](https://www.pywhy.org/econml/_autosummary/econml.dr.SparseLinearDRLearner.html) : Official documentation for EconML's SparseLinearDRLearner.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LogisticRegression, LassoCV

    from caml.estimators.dr import WrappedSparseLinearDRLearner
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

    # Generate synthetic data with many features
    gen = SyntheticDataGenerator(
        n_cont_modifiers=10,
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

    # Fit sparse linear DR learner
    estimator = WrappedSparseLinearDRLearner(
        model_propensity=LogisticRegression(),
        model_regression=LassoCV(),
        alpha=0.1,
        cv=3,
        random_state=42
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE (sparse feature selection applied)
    cate = estimator.effect(data.X)
    effect_inference = estimator.effect_inference(data.X)

    print(f"Mean CATE: {cate.mean():.3f}")
    print(f"Mean StdErr: {effect_inference.stderr.mean():.3f}")
    ```
    """

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.ANALYTIC, InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.CATE,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=True,
        supports_weights=True,
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=True,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_propensity", model_type="treatment"),
        NuisanceModelSpec(name="model_regression", model_type="regression"),
        # CategoricalSpec(
        #     name="featurizer",
        #     choices=[
        #         None,
        #         RobustScaler(),  # Just scaling (no polynomials)
        #         Pipeline(
        #             [
        #                 ("scaler", RobustScaler()),
        #                 ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        #             ]
        #         ),
        #         Pipeline(
        #             [
        #                 ("scaler", RobustScaler()),
        #                 (
        #                     "poly",
        #                     PolynomialFeatures(
        #                         degree=2, interaction_only=True, include_bias=False
        #                     ),
        #                 ),
        #             ]
        #         ),
        #     ],
        # ),
        CategoricalSpec(name="alpha", choices=["auto", 0.01, 0.05, 0.1, 0.5, 1.0]),
        IntSpec(name="n_alphas", lower=50, upper=150, step=50),
        CategoricalSpec(name="alpha_cov", choices=["auto", 0.01, 0.1, 1.0]),
        IntSpec(name="n_alphas_cov", lower=5, upper=15, step=5),
        BoolSpec(name="fit_cate_intercept"),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
        ConstantSpec(name="max_iter", value=1000),
        ConstantSpec(name="tol", value=1e-4),
        FloatSpec(name="min_propensity", lower=1e-6, upper=0.01, log=True),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = SparseLinearDRLearner(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedSparseLinearDRLearner:
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
        SparseLinearDRLearner automatically handles discrete treatments and provides
        analytic inference via debiased Lasso.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
            sample_weight=data.weights if data.weights is not None else None,
            **fit_kwargs,
        )

        self._is_fitted = True

        return self


@auto_register(name="ForestDRLearner", family="dr")
class WrappedForestDRLearner(BaseEconMLWrapperMixin):
    """Wrapper for EconML's ForestDRLearner estimator.

    ForestDRLearner estimates CATE using doubly robust learning with a random forest
    final model for nonparametric heterogeneity estimation. Provides bootstrap confidence
    intervals. Supports binary and multi-valued treatments with continuous outcomes.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dr.ForestDRLearner``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for AutoML tuning of this estimator.

    See Also
    --------
    [EconML ForestDRLearner](https://www.pywhy.org/econml/_autosummary/econml.dr.ForestDRLearner.html) : Official documentation for EconML's ForestDRLearner.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LogisticRegression, LassoCV

    from caml.estimators.dr import WrappedForestDRLearner
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

    # Fit forest DR learner
    estimator = WrappedForestDRLearner(
        model_propensity=LogisticRegression(),
        model_regression=LassoCV(),
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        cv=3,
        random_state=42
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE (nonparametric heterogeneity)
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")
    print(f"CATE Range: [{cate.min():.3f}, {cate.max():.3f}]")
    ```
    """

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.BOOTSTRAP},
        estimands={
            Estimand.ATE,
            Estimand.ATT,
            Estimand.ATC,
            Estimand.CATE,
            Estimand.GATE,
        },
        supports_controls_in_first_stage_only=True,
        supports_weights=True,
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=True,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_propensity", model_type="treatment"),
        NuisanceModelSpec(name="model_regression", model_type="regression"),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
        IntSpec(name="n_estimators", lower=40, upper=520, step=40),
        CategoricalSpec(name="max_depth", choices=[None, 2, 3, 5, 10, 15, 20]),
        FloatSpec(name="min_samples_split", lower=1e-5, upper=0.1, log=True),
        FloatSpec(name="min_samples_leaf", lower=1e-5, upper=0.1, log=True),
        CategoricalSpec(name="max_features", choices=["auto", "sqrt", "log2"]),
        CategoricalSpec(name="max_samples", choices=[0.1, 0.2, 0.3, 0.45, 0.5]),
        CategoricalSpec(name="min_balancedness_tol", choices=[0.1, 0.3, 0.45]),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = ForestDRLearner(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedForestDRLearner:
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
        ForestDRLearner automatically handles discrete treatments and provides
        bootstrap inference for the random forest final model.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
            sample_weight=data.weights if data.weights is not None else None,
            **fit_kwargs,
        )

        self._is_fitted = True

        return self
