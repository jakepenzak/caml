"""Nuisance Tuner Specification."""

from dataclasses import dataclass


@dataclass
class NuisanceTunerSpec:
    r"""Specification for nuisance, or first-stage model, tuning.

    Note, sensible defaults will be chosen in `~~auto_cate.AutoCATE` and many
    will be inferred directly from `~~dataset.CausalDataset` specifications
    (e.g., target variable type and objective).

    Parameters
    ----------
    fit_treatment_model
        Whether to fit the treatment model - $\mathbb{E}[T|X,W]$. If None, the
        decision is made from available estimator
        `~~base_estimator.EstimatorCapabilities`.
    fit_outcome_model
        Whether to fit the outcome model - $\mathbb{E}[Y|X,W]$. If None, the
        decision is made from available estimator
        `~~base_estimator.EstimatorCapabilities`.
    fit_regression_model
        Whether to fit the regression model - $\mathbb{E}[Y|T,X,W]$. If None,
        the decision is made from available estimator
        `~~base_estimator.EstimatorCapabilities`.
    treatment_model_config
        Configuration dictionary of [FLAML AutoML](https://microsoft.github.io/FLAML/docs/reference/automl/automl) kwarg overrides for the treatment model.
    outcome_model_config
        Configuration dictionary of [FLAML AutoML](https://microsoft.github.io/FLAML/docs/reference/automl/automl) kwarg overrides for the outcome model.
    regression_model_config
        Configuration dictionary of [FLAML AutoML](https://microsoft.github.io/FLAML/docs/reference/automl/automl) kwarg overrides for the regression model.

    See Also
    --------
    [FLAML AutoML](https://microsoft.github.io/FLAML/docs/reference/automl/automl) : Available keyword arguments for FLAML AutoML.

    Examples
    --------
    ```{python}
    from caml.nuisance import NuisanceTunerSpec


    spec = NuisanceTunerSpec(
        fit_treatment_model=True,
        fit_outcome_model=True,
        fit_regression_model=False,
        treatment_model_config={"estimator_list": ["lgbm", "rf"], "use_ray": True},
        outcome_model_config={"estimator_list": ["extra_tree", "xgb_limitdepth"], "use_ray": True},
    )
    ```
    """

    fit_treatment_model: bool | None = None
    fit_outcome_model: bool | None = None
    fit_regression_model: bool | None = None

    treatment_model_config: dict | None = None
    outcome_model_config: dict | None = None
    regression_model_config: dict | None = None
