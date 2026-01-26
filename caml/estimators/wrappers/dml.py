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

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators.base import BaseWrapperMixin, EstimatorCapabilities
from caml.inference import InferenceType


class WrappedLinearDML(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("LinearDML") (class attribute).

    See Also
    --------
    [EconML LinearDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html#econml.dml.LinearDML) : Official documentation for EconML's LinearDML.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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
    clean_name: str = "LinearDML"
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

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = LinearDML(**self._econml_kwargs)
        self._is_fitted = False

    def fit(
        self,
        data: CausalDataset,
        **fit_kwargs,
    ) -> WrappedLinearDML:
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedSparseLinearDML(BaseWrapperMixin):
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
    clean_name : str
        Human-readable tests/caml/estimators/wrappers/test_orf.pyname for the estimator ("SparseLinearDML").

    See Also
    --------
    [EconML SparseLinearDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.SparseLinearDML.html) : Official documentation for EconML's SparseLinearDML.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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
    clean_name: str = "SparseLinearDML"
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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedCausalForestDML(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("CausalForestDML").

    See Also
    --------
    [EconML CausalForestDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html) : Official documentation for EconML's CausalForestDML.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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
    clean_name: str = "CausalForestDML"
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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedNonParamDML(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("NonParamDML").

    See Also
    --------
    [EconML NonParamDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.NonParamDML.html) : Official documentation for EconML's NonParamDML.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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
    clean_name: str = "NonParamDML"
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

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedKernelDML(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("KernelDML").

    See Also
    --------
    [EconML KernelDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.KernelDML.html) : Official documentation for EconML's KernelDML.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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
    clean_name: str = "KernelDML"
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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
