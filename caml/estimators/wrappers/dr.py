"""Wrappers for EconML's Doubly Robust (DR) learners (e.g., DML under the Interactive Regression Model regime).

Wraps `DRLearner`, `LinearDRLearner`, `SparseLinearDRLearner`, and `ForestDRLearner`
to implement CaML's `AutoCateEstimator` and `InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators import EstimatorCapabilities
from caml.estimators.base import BaseWrapperMixin
from caml.inference import InferenceType


class WrappedDRLearner(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("DRLearner").

    See Also
    --------
    [EconML DRLearner](https://www.pywhy.org/EconML/_autosummary/econml.dr.DRLearner.html) : Official documentation for EconML's DRLearner.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = DRLearner(**self._econml_kwargs)
        self._is_fitted = False

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """Metadata describing the estimator's supported treatment/outcome types, estimands, and inference methods."""
        return EstimatorCapabilities(
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

    @property
    def clean_name(self) -> str:
        """Human-readable name for the estimator."""
        return "DRLearner"

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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedLinearDRLearner(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("LinearDRLearner").

    See Also
    --------
    [EconML LinearDRLearner](https://www.pywhy.org/econml/_autosummary/econml.dr.LinearDRLearner.html) : Official documentation for EconML's LinearDRLearner.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = LinearDRLearner(**self._econml_kwargs)
        self._is_fitted = False

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """Metadata describing the estimator's supported treatment/outcome types, estimands, and inference methods."""
        return EstimatorCapabilities(
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

    @property
    def clean_name(self) -> str:
        """Human-readable name for the estimator."""
        return "LinearDRLearner"

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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedSparseLinearDRLearner(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("SparseLinearDRLearner").

    See Also
    --------
    [EconML SparseLinearDRLearner](https://www.pywhy.org/econml/_autosummary/econml.dr.SparseLinearDRLearner.html) : Official documentation for EconML's SparseLinearDRLearner.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

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

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = SparseLinearDRLearner(**self._econml_kwargs)
        self._is_fitted = False

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """Metadata describing the estimator's supported treatment/outcome types, estimands, and inference methods."""
        return EstimatorCapabilities(
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

    @property
    def clean_name(self) -> str:
        """Human-readable name for the estimator."""
        return "SparseLinearDRLearner"

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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class WrappedForestDRLearner(BaseWrapperMixin):
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
    clean_name : str
        Human-readable name for the estimator ("ForestDRLearner").

    See Also
    --------
    [EconML ForestDRLearner](https://www.pywhy.org/econml/_autosummary/econml.dr.ForestDRLearner.html) : Official documentation for EconML's ForestDRLearner.

    [`BaseWrapperMixin`](base.qmd#caml.estimators.base.BaseWrapperMixin) : Mixin providing common wrapper functionality.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol this wrapper implements.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LogisticRegression, LassoCV

    from caml.estimators.wrappers.dr import WrappedForestDRLearner
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

    def __init__(self, **econml_kwargs):
        self._econml_kwargs = econml_kwargs
        self._estimator = ForestDRLearner(**self._econml_kwargs)
        self._is_fitted = False

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """Metadata describing the estimator's supported treatment/outcome types, estimands, and inference methods."""
        return EstimatorCapabilities(
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

    @property
    def clean_name(self) -> str:
        """Human-readable name for the estimator."""
        return "ForestDRLearner"

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
        self.check_compatibility(data, raise_error=True)

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

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
