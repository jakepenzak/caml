"""Wrappers for EconML's Orthogonal Random Forest estimators.

Wraps `~~econml.orf.DMLOrthoForest` and `~~econml.orf.DROrthoForest` to
implement CaML's `~~base_estimator.AutoCateEstimator` and
`~~base_estimator.InferenceProvider` protocols.
"""

from __future__ import annotations

from econml.orf import DMLOrthoForest, DROrthoForest

from caml.automl import (
    BoolSpec,
    CategoricalSpec,
    FloatSpec,
    IntSpec,
    NuisanceModelSpec,
    SearchSpace,
)
from caml.data.data_enums import Estimand, OutcomeType, TreatmentType
from caml.data.dataset import CausalDataset
from caml.inference.inference_enums import InferenceType
from caml.registry.registry import auto_register

from ..base_estimator import EstimatorCapabilities
from .base_wrapper import BaseEconMLWrapperMixin


@auto_register(name="DMLOrthoForest", family="orf")
class WrappedDMLOrthoForest(BaseEconMLWrapperMixin):
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
        Keyword arguments passed directly to `~~econml.orf.DMLOrthoForest`.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for tuning the estimator.

    See Also
    --------
    `~~econml.orf.DMLOrthoForest` : Official documentation for EconML's DMLOrthoForest.

    Examples
    --------
    ```{python}
    #| echo: false

    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    from caml.estimators.orf import WrappedDMLOrthoForest
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.utilities.synthetic_data import SyntheticDataGenerator
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
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={
            TreatmentType.BINARY,
            TreatmentType.CONTINUOUS,
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
        supports_controls_in_first_stage_only=True,
        supports_weights=False,
        requires_treatment_model=True,
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="model_T", model_type="treatment"),
        NuisanceModelSpec(name="model_Y", model_type="outcome"),
        IntSpec(name="n_trees", lower=50, upper=500, step=50),
        IntSpec(name="min_leaf_size", lower=5, upper=105, step=10),
        CategoricalSpec(name="max_depth", choices=[2, 3, 5, 10, 15, 20, 30]),
        FloatSpec(name="subsample_ratio", lower=0.1, upper=1.0, step=0.1),
        CategoricalSpec(name="bootstrap", choices=[False, True]),
        FloatSpec(name="lambda_reg", lower=0.001, upper=0.1, log=True),
        BoolSpec(name="global_residualization"),
        CategoricalSpec(name="global_res_cv", choices=[2, 3, 5]),
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
            Additional keyword arguments passed to
            `~~econml.orf.DMLOrthoForest.fit()` method.

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
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

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


@auto_register(name="DROrthoForest", family="orf")
class WrappedDROrthoForest(BaseEconMLWrapperMixin):
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
        Keyword arguments passed directly to `~~econml.orf.DROrthoForest`.

    Attributes
    ----------
    capabilities : EstimatorCapabilities
        Metadata describing the estimator's supported treatment/outcome types,
        estimands, and inference methods.
    default_search_space : SearchSpace
        Default hyperparameter search space for tuning the estimator.

    See Also
    --------
    `~~econml.orf.DROrthoForest` : Official documentation for EconML's DROrthoForest.

    Examples
    --------
    ```{python}
    #| echo: false
    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    from caml.estimators.orf import WrappedDROrthoForest
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.utilities.synthetic_data import SyntheticDataGenerator
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
    capabilities: EstimatorCapabilities = EstimatorCapabilities(
        treatment_types={TreatmentType.BINARY, TreatmentType.MULTI},
        outcome_types={OutcomeType.CONTINUOUS},
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

    default_search_space: SearchSpace = (
        NuisanceModelSpec(name="propensity_model", model_type="treatment"),
        NuisanceModelSpec(name="model_Y", model_type="outcome"),
        IntSpec(name="n_trees", lower=50, upper=500, step=50),
        IntSpec(name="min_leaf_size", lower=5, upper=105, step=10),
        CategoricalSpec(name="max_depth", choices=[2, 3, 5, 10, 15, 20, 30]),
        FloatSpec(name="subsample_ratio", lower=0.1, upper=1.0, step=0.1),
        FloatSpec(name="lambda_reg", lower=0.001, upper=0.1, log=True),
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
            Additional keyword arguments passed to
            `~~econml.orf.DROrthoForest.fit()` method.

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
        if not self.is_compatible_with(data):
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with the provided data. "
            )

        self._estimator.fit(
            Y=data.Y,
            T=data.T,
            X=data.X if data.X.size > 0 else None,
            W=data.W if data.W is not None and data.W.size > 0 else None,
        )

        self._is_fitted = True

        return self
