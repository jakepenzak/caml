"""FLAML-based nuisance model tuner."""

from __future__ import annotations

import numpy as np
from flaml import AutoML

from caml.data import CausalDataset

from .spec import NuisanceTunerSpec


class NuisanceTuner:
    r"""FLAML-based nuisance model tuner.

    Parameters
    ----------
    time_budget
        Time budget in seconds for each nuisance model tuning. Default is 300 seconds.
    use_ray
        Whether to use Ray for distributed tuning. Default is False.
    use_spark
        Whether to use Spark for distributed tuning. Default is False.
    seed
        Random seed for reproducibility. Default is None.
    verbose
        Verbosity level for FLAML logging. Default is 0 (silent).

    Attributes
    ----------
    treatment_model_
        Fitted treatment model (propensity score model) after calling `fit()`, if applicable.
    outcome_model_
        Fitted outcome model after calling `fit()`, if applicable.
    regression_model_
        Fitted regression model after calling `fit()`, if applicable.

    Examples
    --------
    ```{python}
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.nuisance import NuisanceTunerSpec, NuisanceTuner

    gen = SyntheticDataGenerator(n_cont_modifiers=3, seed=42)

    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )


    spec = NuisanceTunerSpec(
        fit_treatment_model=True,
        fit_outcome_model=True,
        fit_regression_model=True
    )

    tuner = NuisanceTuner(time_budget=1, verbose=3)

    tuner.fit(data, spec)


    print(f"\n{tuner.treatment_model_}")
    print(f"\n{tuner.outcome_model_}")
    print(f"\n{tuner.regression_model_}")
    ```
    """

    def __init__(
        self,
        time_budget: int = 300,
        use_ray: bool = False,
        use_spark: bool = False,
        seed: int | None = None,
        verbose: int = 0,
    ):
        self.time_budget = time_budget
        self.use_ray = use_ray
        self.use_spark = use_spark
        self.seed = seed
        self.verbose = verbose

        # Fitted models (stored after fit())
        self.treatment_model_ = None
        self.outcome_model_ = None
        self.regression_model_ = None

    def fit(self, data: CausalDataset, spec: NuisanceTunerSpec):
        """Fit nuisance models based on the dataset and nuisance tuner spec.

        Parameters
        ----------
        data
            CausalDataset containing the data for fitting.
        spec
            NuisanceTunerSpec specifying which models to fit and any configuration overrides.
        """
        base_config = self._build_base_config()

        if spec.fit_treatment_model:
            config = self._build_treatment_model_config(data, base_config, spec)
            self.treatment_model_ = self._run_flaml(config)

        if spec.fit_outcome_model:
            config = self._build_outcome_model_config(data, base_config, spec)
            self.outcome_model_ = self._run_flaml(config)

        if spec.fit_regression_model:
            config = self._build_regression_model_config(data, base_config, spec)
            self.regression_model_ = self._run_flaml(config)

    def _build_base_config(self) -> dict:
        """Base FLAML settings."""
        config = {
            "n_jobs": -1,
            "time_budget": self.time_budget,
            "seed": self.seed,
            "verbose": self.verbose,
            "log_file_name": "",
            "early_stop": True,
            "eval_method": "cv",
            "n_splits": 3,
            "starting_points": "static",
            "estimator_list": "auto",
            "retrain_full": False,
        }

        if self.use_ray:
            config["use_ray"] = True
            config["n_concurrent_trials"] = 4
        elif self.use_spark:
            config["use_spark"] = True
            config["n_concurrent_trials"] = 4

        return config

    def _build_treatment_model_config(
        self, data: CausalDataset, base: dict, spec: NuisanceTunerSpec
    ) -> dict:
        """Build config for treatment model E[T|X,W]."""
        config = base.copy()

        # Determine task type
        if data.treatment_type.is_discrete():
            config["task"] = "classification"
            config["metric"] = "log_loss"
        else:
            config["task"] = "regression"
            config["metric"] = "mse"

        # Prepare data
        XW = np.hstack([data.X, data.W]) if data.W is not None else data.X
        config["X_train"] = XW
        config["y_train"] = data.T

        # Apply user overrides
        if spec.treatment_model_config:
            config.update(spec.treatment_model_config)

        return config

    def _build_outcome_model_config(
        self, data: CausalDataset, base: dict, spec: NuisanceTunerSpec
    ) -> dict:
        """Build config for outcome model E[Y|X,W]."""
        config = base.copy()

        # Determine task type
        if data.outcome_type.is_discrete():
            config["task"] = "classification"
            config["metric"] = "log_loss"
        else:
            config["task"] = "regression"
            config["metric"] = "mse"

        # Prepare data
        XW = np.hstack([data.X, data.W]) if data.W is not None else data.X
        config["X_train"] = XW
        config["y_train"] = data.Y

        # Apply user overrides
        if spec.outcome_model_config:
            config.update(spec.outcome_model_config)

        return config

    def _build_regression_model_config(
        self, data: CausalDataset, base: dict, spec: NuisanceTunerSpec
    ) -> dict:
        """Build config for regression model E[Y|X,W,T]."""
        config = base.copy()

        # Task type same as outcome
        if data.outcome_type.is_discrete():
            config["task"] = "classification"
            config["metric"] = "log_loss"
        else:
            config["task"] = "regression"
            config["metric"] = "mse"

        # Prepare data (include treatment)
        XWT = (
            np.hstack([data.X, data.W, data.T])
            if data.W is not None
            else np.hstack([data.X, data.T])
        )
        config["X_train"] = XWT
        config["y_train"] = data.Y

        # Apply user overrides
        if spec.regression_model_config:
            config.update(spec.regression_model_config)

        return config

    def _run_flaml(self, config: dict):
        """Run AutoML and return best estimator."""
        automl = AutoML()
        automl.fit(**config)
        return automl.model.estimator  # pyright: ignore[reportOptionalMemberAccess]
