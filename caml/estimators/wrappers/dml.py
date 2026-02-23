"""Wrappers for EconML's Double Machine Learning estimators under the Partially Linear Model regime.

Wraps `LinearDML`, `SparseLinearDML`, `CausalForestDML`, `NonParamDML`, and `KernelDML`
to implement CaML's `AutoCateEstimator` and `InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.dml import (
    CausalForestDML,
    KernelDML,
    LinearDML,
    NonParamDML,
    SparseLinearDML,
)
from sklearn.linear_model import LinearRegression

from caml.automl import (
    BoolSpec,
    CategoricalSpec,
    ConstantSpec,
    FloatSpec,
    IntSpec,
    NuisanceModelSpec,
    SearchSpace,
)
from caml.automl.search_space import StandardMLSpec
from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.inference import InferenceType
from caml.registry import auto_register

from ..base_estimator import EstimatorCapabilities
from .base_wrapper import BaseEconMLWrapperMixin


@auto_register(name="LinearDML", family="dml")
class WrappedLinearDML(BaseEconMLWrapperMixin):
    """Wrapper for EconML's LinearDML estimator.

    LinearDML estimates CATE using Double Machine Learning with a linear final model.
    Supports binary, multi-valued, and continuous treatments with continuous outcomes.
    Provides analytic confidence intervals via debiased moment conditions.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dml.LinearDML``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods (class attribute).
    default_search_space : SearchSpace
        Default hyperparameter search space for tuning the estimator (class attribute).

    See Also
    --------
    [EconML LinearDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html#econml.dml.LinearDML) : Official documentation for EconML's LinearDML.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from caml.estimators.dml import WrappedLinearDML
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
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

    # Fit estimator
    estimator = WrappedLinearDML(model_y=LinearRegression(), model_t=LogisticRegression(), cv=3, random_state=42)
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")

    # Get confidence intervals
    effect_inference = estimator.effect_inference(data.X)

    print(f"Mean CATE: {effect_inference.effect.mean():.3f}")
    print(f"Mean StdErr: {effect_inference.stderr.mean():.3f}")
    ```
    """

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.CONTINUOUS,
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
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_y", model_type="outcome"),
        NuisanceModelSpec(name="model_t", model_type="treatment"),
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
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = LinearDML(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedLinearDML:
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )
        self._estimator.discrete_treatment = (
            True if data.treatment_type.is_discrete() else False
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


@auto_register(name="SparseLinearDML", family="dml")
class WrappedSparseLinearDML(BaseEconMLWrapperMixin):
    """Wrapper for EconML's SparseLinearDML estimator.

    SparseLinearDML estimates CATE using Double Machine Learning with a sparse linear (Lasso)
    final model for feature selection. Supports binary, multi-valued, and continuous treatments
    with continuous outcomes. Provides analytic confidence intervals via debiased Lasso.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dml.SparseLinearDML``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for tuning the estimator.

    See Also
    --------
    [EconML SparseLinearDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.SparseLinearDML.html) : Official documentation for EconML's SparseLinearDML.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from caml.estimators.dml import WrappedSparseLinearDML
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

    # Generate synthetic data with many features
    gen = SyntheticDataGenerator(
        n_cont_modifiers=10,  # Many features for sparsity
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

    # Fit estimator with sparse regularization
    estimator = WrappedSparseLinearDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(),
        alpha=0.1,  # Lasso regularization strength
        cv=3,
        random_state=42
    )
    estimator.fit(data)

    # Ensure it satisfies protocol
    assert isinstance(estimator, AutoCateEstimator)

    # Predict CATE (sparse feature selection applied)
    cate = estimator.effect(data.X)
    print(f"Mean CATE: {cate.mean():.3f}")

    # Get confidence intervals
    effect_inference = estimator.effect_inference(data.X)
    print(f"Mean StdErr: {effect_inference.stderr.mean():.3f}")
    ```
    """

    # Class attributes
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.CONTINUOUS,
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
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_y", model_type="outcome"),
        NuisanceModelSpec(name="model_t", model_type="treatment"),
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
        FloatSpec(name="alpha", lower=1e-4, upper=10.0, log=True),
        IntSpec(name="n_alphas", lower=50, upper=150, step=50),
        CategoricalSpec(name="alpha_cov", choices=["auto", 0.01, 0.1, 1.0]),
        IntSpec(name="n_alphas_cov", lower=5, upper=15, step=5),
        BoolSpec(name="fit_cate_intercept"),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
        ConstantSpec(name="max_iter", value=1000),
        ConstantSpec(name="tol", value=1e-4),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = SparseLinearDML(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedSparseLinearDML:
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
        This method automatically sets ``discrete_treatment`` and ``discrete_outcome``
        flags based on the data's treatment_type and outcome_type.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )
        self._estimator.discrete_treatment = (
            True if data.treatment_type.is_discrete() else False
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


@auto_register(name="CausalForestDML", family="dml")
class WrappedCausalForestDML(BaseEconMLWrapperMixin):
    """Wrapper for EconML's CausalForestDML estimator.

    CausalForestDML estimates CATE using Double Machine Learning with a causal forest
    final model. The causal forest provides nonparametric heterogeneous treatment effect
    estimation. Supports binary, multi-valued, and continuous treatments with continuous
    outcomes. Provides bootstrap confidence intervals.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dml.CausalForestDML``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for tuning the estimator.

    See Also
    --------
    [EconML CausalForestDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html) : Official documentation for EconML's CausalForestDML.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    from caml.estimators.dml import WrappedCausalForestDML
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.estimators import AutoCateEstimator

    # Generate synthetic data with nonlinear effects
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

    # Fit causal forest
    estimator = WrappedCausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50),
        model_t=RandomForestClassifier(n_estimators=50),
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
            TreatmentType.CONTINUOUS,
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
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_y", model_type="outcome"),
        NuisanceModelSpec(name="model_t", model_type="treatment"),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
        IntSpec(name="n_estimators", lower=40, upper=520, step=40),
        CategoricalSpec(name="criterion", choices=["mse", "het"]),
        CategoricalSpec(name="max_depth", choices=[None, 2, 3, 5, 10, 15, 20]),
        FloatSpec(name="min_samples_split", lower=1e-5, upper=0.1, log=True),
        FloatSpec(name="min_samples_leaf", lower=1e-5, upper=0.1, log=True),
        CategoricalSpec(name="min_var_fraction_leaf", choices=[None, 0.01, 0.05, 0.1]),
        CategoricalSpec(name="max_features", choices=["auto", "sqrt", "log2"]),
        CategoricalSpec(name="max_samples", choices=[0.1, 0.2, 0.3, 0.45, 0.5]),
        CategoricalSpec(name="min_balancedness_tol", choices=[0.1, 0.3, 0.45]),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = CausalForestDML(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedCausalForestDML:
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
        This method automatically sets ``discrete_treatment`` flag based on the
        data's treatment_type. CausalForestDML only supports continuous outcomes.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )
        self._estimator.discrete_treatment = (
            True if data.treatment_type.is_discrete() else False
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


@auto_register(name="NonParamDML", family="dml")
class WrappedNonParamDML(BaseEconMLWrapperMixin):
    """Wrapper for EconML's NonParamDML estimator.

    NonParamDML estimates CATE using Double Machine Learning with a fully nonparametric
    final model. Supports binary and continuous treatments with continuous outcomes.
    Provides bootstrap confidence intervals only (no analytic inference).

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dml.NonParamDML``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for tuning the estimator.

    See Also
    --------
    [EconML NonParamDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.NonParamDML.html) : Official documentation for EconML's NonParamDML.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    from caml.estimators.dml import WrappedNonParamDML
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

    # Fit with nonparametric final model
    estimator = WrappedNonParamDML(
        model_y=GradientBoostingRegressor(n_estimators=50),
        model_t=GradientBoostingRegressor(n_estimators=50),
        model_final=RandomForestRegressor(n_estimators=100, max_depth=5),
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
            TreatmentType.CONTINUOUS,
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
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_y", model_type="outcome"),
        NuisanceModelSpec(name="model_t", model_type="treatment"),
        StandardMLSpec(name="model_final"),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        if "model_y" not in self._econml_kwargs:
            self._econml_kwargs["model_y"] = LinearRegression()
        if "model_t" not in self._econml_kwargs:
            self._econml_kwargs["model_t"] = LinearRegression()
        if "model_final" not in self._econml_kwargs:
            self._econml_kwargs["model_final"] = LinearRegression()
        self._estimator = NonParamDML(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedNonParamDML:
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
        This method automatically sets ``discrete_treatment`` flag based on the
        data's treatment_type. NonParamDML only supports continuous outcomes.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )
        self._estimator.discrete_treatment = (
            True if data.treatment_type.is_discrete() else False
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


@auto_register(name="KernelDML", family="dml")
class WrappedKernelDML(BaseEconMLWrapperMixin):
    """Wrapper for EconML's KernelDML estimator.

    KernelDML estimates CATE using Double Machine Learning with kernel methods for the
    final model. Uses kernel ridge regression for nonparametric CATE estimation. Supports
    binary and continuous treatments with continuous outcomes. Provides bootstrap confidence
    intervals only.

    *Note: All attributes and methods on the underlying EconML estimator are accessible
    via this wrapper through delegation, if not explicitly overridden.*

    Parameters
    ----------
    **econml_kwargs
        Keyword arguments passed directly to ``econml.dml.KernelDML``.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.

    See Also
    --------
    [EconML KernelDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.KernelDML.html) : Official documentation for EconML's KernelDML.

    Examples
    --------
    ```{python}
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    from caml.estimators.dml import WrappedKernelDML
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

    # Fit with kernel method
    estimator = WrappedKernelDML(
        model_y=GradientBoostingRegressor(),
        model_t=GradientBoostingClassifier(),
        cv=3,
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
            TreatmentType.CONTINUOUS,
            TreatmentType.MULTI,
        },
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        inference_types={InferenceType.BOOTSTRAP, InferenceType.ANALYTIC},
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
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_y", model_type="outcome"),
        NuisanceModelSpec(name="model_t", model_type="treatment"),
        IntSpec(name="dim", lower=10, upper=100, step=10),
        CategoricalSpec(name="bw", choices=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
        BoolSpec(name="fit_cate_intercept"),
        IntSpec(name="cv", lower=2, upper=5, step=1),
        CategoricalSpec(name="mc_iters", choices=[None, 2, 3]),
        CategoricalSpec(name="mc_agg", choices=["mean", "median"]),
    )

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = KernelDML(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedKernelDML:
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
        This method automatically sets ``discrete_treatment`` flag based on the
        data's treatment_type. KernelDML only supports continuous outcomes.
        """
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.discrete_outcome = (
            True if data.outcome_type.is_discrete() else False
        )
        self._estimator.discrete_treatment = (
            True if data.treatment_type.is_discrete() else False
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
