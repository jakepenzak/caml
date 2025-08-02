from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml import dml, dr, metalearners
from econml._ortho_learner import _OrthoLearner
from econml.dml.dml import NonParamDML
from econml.inference import BootstrapInference
from econml.score import EnsembleCateEstimator, RScorer
from joblib import Parallel, delayed

from caml.core._base import BaseCamlEstimator
from caml.core.modeling.model_bank import (
    AutoCateEstimator,
    available_estimators,
)
from caml.generics import logging as clg
from caml.generics.decorators import experimental, narrate, timer
from caml.generics.interfaces import FittedAttr, PandasConvertibleDataFrame
from caml.generics.logging import DEBUG, INFO, WARNING
from caml.generics.monkey_patch import DRTester
from caml.generics.utils import is_module_available

_HAS_PYSPARK = is_module_available("pyspark")
_HAS_RAY = is_module_available("ray")

if _HAS_RAY:
    import ray

if TYPE_CHECKING:
    import ray
    from econml._cate_estimator import BaseCateEstimator
    from econml.inference import PopulationSummaryResults
    from econml.inference._inference import InferenceResults
    from sklearn.base import BaseEstimator


warnings.filterwarnings("ignore")


# TODO: Refactor all docstrings!!
@experimental
class AutoCATE(BaseCamlEstimator):
    r"""The AutoCATE class is an high-level API facilitating an AutoML framework for CATE estimation, built on top of the EconML library.

    **AutoCATE is experimental and may change significantly in future versions.**

    The CATE is defined as $\mathbb{E}[\tau|\mathbf{X}]$
    where $\tau$ is the treatment effect and $\mathbf{X}$ is the set of covariates.

    This class is built on top of the EconML library and provides a high-level API for AutoML for fitting, validating, and making predictions/inference with CATE models,
    with best practices built directly into the API. The class is designed to be easy to use and understand, while still providing flexibility for advanced users.

    Note that first-stage models are estimated using a full AutoML framework via [Flaml](https://microsoft.github.io/FLAML/), whereas the second-stage models
    are currently estimated & selected based on pre-specified set of models (or custom CATE models) passed - there is no tuning of hyperparameters. This is
    on the roadmap for future versions of AutoCATE.

    For technical details on the AutoCATE class, see here.

    **Note**: All the standard assumptions of Causal Inference apply to this class (e.g., exogeneity/unconfoundedness, overlap, positivity, etc.).
        The class does not check for these assumptions and assumes that the user has already thought through these assumptions before using the class.

    For outcome/treatment support, see [matrix](support_matrix.qmd).

    For a more detailed working example, see [AutoCATE Example](../03_Examples/AutoCATE.qmd).

    Parameters
    ----------
    Y : str
        The str representing the column name for the outcome variable.
    T : str
        The str representing the column name(s) for the treatment variable(s).
    X : Sequence[str]
        The sequence of feature names representing the feature set to be utilized for estimating heterogeneity/CATE.
    W : Sequence[str] | None
        The sequence of feature names representing the confounder/control feature set to be utilized only for nuisance function estimation. When W is passed, only Orthogonal learners will be leveraged.
    discrete_treatment : bool
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome : bool
        A boolean indicating whether the outcome is binary or continuous.
    model_Y : dict | BaseEstimator | None
        A dictionary of [Flaml](https://microsoft.github.io/FLAML/docs/reference/automl/automl) kwarg overrides or a BaseEstimator instance for the outcome model - $\mathbb{E}[Y | \mathbf{X},\mathbf{W}]$.
    model_T : dict | BaseEstimator | None
        A dictionary of [Flaml](https://microsoft.github.io/FLAML/docs/reference/automl/automl) kwarg overrides or a BaseEstimator instance for the treatment/propensity model - $\mathbb{E}[T | \mathbf{X},\mathbf{W}]$.
    model_regression : dict | BaseEstimator | None
        A dictionary of [Flaml](https://microsoft.github.io/FLAML/docs/reference/automl/automl) kwarg overrides or a BaseEstimator instance for the regression model - $\mathbb{E}[Y | \mathbf{X},\mathbf{W},T]$.
    enable_categorical : bool
        A boolean indicating whether to enable categorical encoding for the models. When set to True, pandas categorical types will be converted to ordinal encodings. For one-hot encoding, please implement it yourself.
    n_jobs : int
        The number of jobs to run in parallel for model training using joblib.
    use_ray : bool
        A boolean indicating whether to use Ray for distributed computing, utilized in both AutoML for first-stage models and AutoML for CATE models. This argument overrides the n_jobs parameter.
    ray_remote_func_options_kwargs : dict | None
        A dictionary of Ray remote function options for distributed computing. See [here](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) for options.
    use_spark : bool
        A boolean indicating whether to use Spark for distributed computing, utilized only in AutoML for first-stage models.
    verbose : int
        The verbosity level. 0 = NOTSET, 1 = DEBUG, 2 = INFO, 3 = WARNING, 4 = ERROR, 5 = CRITICAL
    seed : int | None
        The seed to use for the random number generator.

    Attributes
    ----------
    Y : list[str]
        The list of str representing the column names for the outcome variable $Y$.
    T : list[str]
        The list of str representing the column names for the treatment variable $T$.
    X : list[str]
        The list of str representing the confounder/control feature set $\mathbf{X}$ to be utilized for estimating heterogeneity/CATE and nuisance function estimation where applicable.
    W : list[str] | None
        The list of str representing the confounder/control feature set $\mathbf{W}$ to be utilized only for nuisance function estimation, where applicable. These will be included by default in Meta-Learners.
    available_estimators : list[str]
        A list of the available CATE estimators out of the box. Validity of estimator at runtime will depend on the outcome and treatment types and be automatically selected.
    model_Y: BaseEstimator
        The selected outcome model - $\mathbb{E}[Y|\mathbf{X},\mathbf{W}]$.
    model_T: BaseEstimator
        The selected treatment model - $\mathbb{E}[T|\mathbf{X},\mathbf{W}]$.
    model_regression: BaseEstimator
        The selected regression model - $\mathbb{E}[Y|\mathbf{X},\mathbf{W},T]$.
    rscores : dict[str, float]
        The dictionary of the Rscores on the validation set for each CATE estimator fitted during model selection.
    test_results : dict[str, float] | EvaluationResults
        The dictionary of the final test results on the test set for the best_estimator selected, if [RScorer](https://www.pywhy.org/EconML/_autosummary/econml.score.RScorer.html) is used, otherwise EvaluationResults returned from [DRTester.evaluate_all](https://www.pywhy.org/EconML/_autosummary/econml.validate.DRTester.html#econml.validate.DRTester.evaluate_all)
    best_estimator : BaseCateEstimator
        The best EconML CATE estimator selected.
    best_estimator_name : str
        The name of the best EconML CATE estimator selected as passed to the AutoCateEstimator constructor.


    Examples
    --------
    ```{python}
    from caml import AutoCATE
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data_generator = SyntheticDataGenerator(seed=10, n_cont_modifiers=1, n_cont_confounders=1)
    df = data_generator.df

    auto_cate = AutoCATE(
        Y="Y1_continuous",
        T="T1_binary",
        X=[c for c in df.columns if "X" in c or "W" in c],
        model_Y={"time_budget": 10},
        model_T={"time_budget": 10},
        model_regression={"time_budget": 10},
        discrete_treatment=True,
        discrete_outcome=False,
    )

    print(auto_cate)
    ```
    """

    best_estimator: BaseCateEstimator | EnsembleCateEstimator = FittedAttr(
        "_best_estimator"
    )  # pyright: ignore[reportAssignmentType]
    best_estimator_name: str = FittedAttr("_best_estimator_name")  # pyright: ignore[reportAssignmentType]
    model_Y: BaseEstimator = FittedAttr("_model_Y")  # pyright: ignore[reportAssignmentType]
    model_T: BaseEstimator = FittedAttr("_model_T")  # pyright: ignore[reportAssignmentType]
    model_regression: BaseEstimator = FittedAttr("_model_regression")  # pyright: ignore[reportAssignmentType]

    def __init__(
        self,
        Y: str,
        T: str,
        X: Sequence[str] | None = None,
        W: Sequence[str] | None = None,
        *,
        discrete_treatment: bool = True,
        discrete_outcome: bool = False,
        model_Y: dict | BaseEstimator | None = None,
        model_T: dict | BaseEstimator | None = None,
        model_regression: dict | BaseEstimator | None = None,
        enable_categorical: bool = False,
        n_jobs: int = 1,
        use_ray: bool = False,
        ray_remote_func_options_kwargs: dict | None = None,
        use_spark: bool = False,
        verbose: int = 2,
        seed: int | None = None,
    ):
        self.verbose = verbose * 10
        clg.configure_logging(level=verbose * 10)
        self.Y = [Y] if isinstance(Y, str) else list(Y)
        self.T = [T] if isinstance(T, str) else list(T)
        self.X = list(X) if X else list()
        self.W = list(W) if W else list()
        self._discrete_treatment = discrete_treatment
        self._discrete_outcome = discrete_outcome
        self._model_Y_specs = model_Y
        self._model_T_specs = model_T
        self._model_regression_specs = model_regression
        self._enable_categorical = enable_categorical
        self._n_jobs = n_jobs
        self._use_ray = use_ray
        self._ray_remote_func_options_kwargs = (
            ray_remote_func_options_kwargs
            if ray_remote_func_options_kwargs is not None
            else {}
        )
        self._use_spark = use_spark
        self._seed = seed
        self.available_estimators = list(available_estimators.keys())

        self._model_T = None
        self._model_Y = None
        self._model_regression = None
        self._bs_estimator = None
        self._fitted = False

        if len(self.W) > 0:
            WARNING(
                "Only Orthogonal Learners are currently supported with 'W', as Meta-Learners neccesitate 'W' in final CATE learner. "
                "If you don't care about 'W' features being used in final CATE model, add it to 'X' argument insead."
            )

        if use_ray and not _HAS_RAY:
            raise ImportError(
                "Ray is not installed. Please install Ray to use it for parallel processing."
            )

        if use_spark and not _HAS_PYSPARK:
            raise ImportError(
                "PySpark is not installed. Please install PySpark optional dependencies via `pip install caml[pyspark]`."
            )

    @narrate(preamble=clg.LOGO, epilogue=None)
    @timer("End-to-end Fitting, Validation, & Testing")
    def fit(
        self,
        df: PandasConvertibleDataFrame,
        cate_estimators: Sequence[str] = list(available_estimators.keys()),
        additional_cate_estimators: Sequence[AutoCateEstimator] = list(),
        ensemble: bool = False,
        refit_final: bool = True,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.1,
        rscorer_kwargs: dict = dict(),
        n_groups_dr_tester=5,
        n_bootstrap_dr_tester=100,
    ):
        """Run end-to-end fitting, validation & model selection, and testing for CATE models.

        Parameters
        ----------
        df : PandasConvertibleDataFrame
            The dataset to fit the CATE model on. Accepts a Pandas DataFrame or a compatible object with a `to_pandas()` or `.toPandas()` method.
        cate_estimators : Sequence[str]
            The out-of-the-box CATE estimators to use. Accessible via `self.available_estimators`.
        additional_cate_estimators : Sequence[AutoCateEstimator]
            Additional CATE estimators to use.
        ensemble : bool
            Whether to use an ensemble of CATE estimators.
        refit_final : bool
            Whether to refit the final CATE estimator on the entire dataset after model selection.
        validation_fraction : float
            The fraction of the dataset to use for validation.
        test_fraction : float
            The fraction of the dataset to use for testing.
        rscorer_kwargs : dict
            Additional keyword arguments to pass to [RScorer](https://www.pywhy.org/EconML/_autosummary/econml.score.RScorer.html).
        n_groups_dr_tester : int
            The number of groups to use for the [DRTester](https://www.pywhy.org/EconML/_autosummary/econml.validate.DRTester.html).
        n_bootstrap_dr_tester : int
            The number of bootstrap samples to use for the [DRTester](https://www.pywhy.org/EconML/_autosummary/econml.validate.DRTester.html).

        Examples
        --------
        ```{python}
        from caml import AutoCateEstimator
        from econml.dml import LinearDML

        my_custom_estimator = AutoCateEstimator(name="MyCustomEstimator",estimator=LinearDML())

        auto_cate.fit(
            df = df,
            cate_estimators = auto_cate.available_estimators,
            additional_cate_estimators = [my_custom_estimator],
        )
        ```
        """
        self._fitted = False
        INFO(f"{self} \n")

        for ce in cate_estimators:
            if ce not in self.available_estimators:
                raise ValueError(f"Invalid cate_estimator: {ce}")

        for ace in additional_cate_estimators:
            if not isinstance(ace, AutoCateEstimator):
                raise ValueError(
                    f"Invalid cate_estimator: {ace}. Must be instance of AutoCateEstimator."
                )

        if self._use_ray:
            if not ray.is_initialized():
                ray.init()

        pd_df = self._convert_dataframe_to_pandas(df=df)
        if self._enable_categorical:
            pd_df, self._categorical_mappings = self._encode_categoricals(
                df=pd_df, is_training=True
            )
        self._find_nuisance_functions(df=pd_df)
        splits = self._split_data(
            df=pd_df,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        fitted_estimators = self._fit_estimators(
            cate_estimators=cate_estimators,
            additional_cate_estimators=additional_cate_estimators,
            splits=splits,
        )
        self._validate(
            fitted_estimators=fitted_estimators,
            splits=splits,
            ensemble=ensemble,
            rscorer_kwargs=rscorer_kwargs,
        )
        self._test(
            splits=splits,
            n_groups=n_groups_dr_tester,
            n_bootstrap=n_bootstrap_dr_tester,
            rscorer_kwargs=rscorer_kwargs,
        )

        self._fitted = True

        if refit_final:
            self.refit_final(df=pd_df)

    @timer("Estimating ATE(s)")
    def estimate_ate(
        self,
        df: PandasConvertibleDataFrame,
        effect_mode: str = "discrete",
        T0: int = 0,
        T1: int = 1,
        T: np.ndarray | pd.DataFrame | None = None,
        return_inference: bool = False,
        n_bootstrap_samples: int = 100,
        alpha: float = 0.05,
        value: int = 0,
    ) -> float | PopulationSummaryResults:
        r"""Calculate the average treatment effect (ATE) $\mathbb{E}[\tau(\mathbf{X})]$.

        This method can be used for group average treatment effects (GATEs) $\mathbb{E}[\tau(\mathbf{X})|G]$ by filtering the input df to select a specific group.

        Two effect modes are supported: `discrete` and `marginal`.

        In `discrete` mode, the effect is calculated between two specific treatment levels T0 and T1 - $\tau(\mathbf{X}, T0, T1)$.

        In `marginal` mode, the effect is calculated for each observation as a gradient around their treatment levels - $\partial_{\tau}(T,\mathbf{x})$

        See [EconML](https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html#econml.dml.LinearDML.__init__) for more details.

        Parameters
        ----------
        df : PandasConvertibleDataFrame
            The data frame containing the data.
        effect_mode : str
            The mode of effect calculation. Can be "marginal" or "discrete".
        T0 : int
            The base treatment level when effect_mode is "discrete".
        T1 : int
            The target treatment level when effect_mode is "discrete".
        T : np.ndarray | pd.DataFrame | None
            The base treatment levels for each observation when effect_mode is "marginal".
        return_inference : bool
            Whether to return inference results.
        alpha : float
            The level of confidence in the reported interval.
        value : int
            The mean value to test under the null hypothesis.

        Returns
        -------
        float | PopulationSummaryResults
            The average treatment effect estimate if return_inference is False. Otherwise, an instance of [PopulationSummaryResults](https://www.pywhy.org/EconML/_autosummary/econml.inference.PopulationSummaryResults.html) is returned.

        Examples
        --------
        ```{python}
        # Return scalar ATE
        auto_cate.estimate_ate(
            df = df,
            effect_mode = "discrete",
            T0 = 0,
            T1 = 1,
        )
        ```
        ```{python}
        # Return ATE with Inference
        auto_cate.estimate_ate(
            df = df,
            effect_mode = "discrete",
            T0 = 0,
            T1 = 1,
            return_inference = True,
            alpha = 0.05,
            value = 0,
        )
        ```
        """
        res = self.estimate_cate(
            df=df,
            effect_mode=effect_mode,
            T0=T0,
            T1=T1,
            T=T,
            return_inference=return_inference,
            n_bootstrap_samples=n_bootstrap_samples,
        )
        if return_inference:
            return res.population_summary(alpha=alpha, value=value)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            ate = np.mean(res)  # pyright: ignore[reportCallIssue, reportArgumentType]
            return ate

    @timer("Estimating CATEs")
    def estimate_cate(
        self,
        df: PandasConvertibleDataFrame,
        effect_mode: str = "discrete",
        T0: int = 0,
        T1: int = 1,
        T: np.ndarray | pd.DataFrame | None = None,
        return_inference: bool = False,
        n_bootstrap_samples: int = 100,
    ) -> np.ndarray | InferenceResults:
        """TODO: Docstring."""
        if effect_mode not in ["discrete", "marginal"]:
            raise ValueError(
                f"Invalid effect_mode: {effect_mode} Must be either 'discrete' or 'marginal'"
            )

        pd_df: pd.DataFrame = self._convert_dataframe_to_pandas(df)
        if self._enable_categorical:
            pd_df, _ = self._encode_categoricals(
                pd_df, categorical_mappings=self._categorical_mappings
            )

        X = pd_df[self.X]
        if T is None:
            T = pd_df[self.T].to_numpy()

        if return_inference:
            if self._best_estimator._inference is None or isinstance(  # pyright: ignore[reportAttributeAccessIssue]
                self._best_estimator, EnsembleCateEstimator
            ):
                WARNING(
                    f"Asymptotic inference is not supported for {self._best_estimator_name}. Falling back to bootstrap. Initial fit will happen once and can be expensive."
                )
                if self._bs_estimator is None:
                    self._bs_estimator = BootstrapInference(
                        n_bootstrap_samples=n_bootstrap_samples, n_jobs=self._n_jobs
                    )

                    self._bs_estimator.fit(
                        self._best_estimator,
                        Y=pd_df[self.Y],
                        T=pd_df[self.T],
                        X=X if not X.empty else None,
                        W=pd_df[self.W] if not pd_df[self.W].empty else None,
                    )

                if effect_mode == "discrete":
                    inference = self._bs_estimator.effect_inference(X, T0=T0, T1=T1)
                else:
                    inference = self._bs_estimator.marginal_effect_inference(T, X)

            elif effect_mode == "discrete":
                inference = self._best_estimator.effect_inference(X, T0=T0, T1=T1)
            else:
                inference = self._best_estimator.marginal_effect_inference(T, X)

            return inference
        else:
            if effect_mode == "discrete":
                cates = self._best_estimator.effect(X, T0=T0, T1=T1)
            else:
                cates = self._best_estimator.marginal_effect(T, X)

            return cates

    def predict(
        self, df: PandasConvertibleDataFrame, **kwargs
    ) -> np.ndarray | InferenceResults:
        """Alias for `estimate_cate`."""
        return self.estimate_cate(df, **kwargs)

    def effect(
        self, df: PandasConvertibleDataFrame, **kwargs
    ) -> np.ndarray | InferenceResults:
        """Alias for `estimate_cate`."""
        return self.estimate_cate(df, **kwargs)

    @narrate(preamble=clg.AUTOML_NUISANCE_PREAMBLE)
    @timer("Nuisance Function AutoML")
    def _find_nuisance_functions(self, df: pd.DataFrame):
        base_settings = {
            "n_jobs": -1,
            "log_file_name": "",
            "seed": self._seed,
            "time_budget": 300,
            "early_stop": "True",
            "eval_method": "cv",
            "n_splits": 3,
            "starting_points": "static",
            "estimator_list": "auto",
            "verbose": 0,
        }

        if self._use_spark:
            base_settings["use_spark"] = True
            base_settings["n_concurrent_trials"] = 4
        elif self._use_ray:
            base_settings["use_ray"] = True
            base_settings["n_concurrent_trials"] = 4

        # Model configurations: (model_name, outcome, features, discrete_outcome)
        model_configs = [
            ("model_Y", self.Y, self.X + self.W, self._discrete_outcome),
            (
                "model_regression",
                self.Y,
                self.X + self.W + list(self.T),
                self._discrete_outcome,
            ),
            ("model_T", self.T, self.X + self.W, self._discrete_treatment),
        ]

        for model_name, outcome, features, discrete_outcome in model_configs:
            flaml_kwargs = base_settings.copy()

            model_arg = getattr(self, f"_{model_name}_specs")

            if isinstance(model_arg, dict):
                flaml_kwargs.update(model_arg)
            elif model_arg is None:
                pass
            else:
                setattr(self, f"_{model_name}", model_arg)
                continue

            flaml_kwargs["label"] = outcome[0]
            flaml_kwargs["dataframe"] = df[features + outcome]

            if discrete_outcome:
                flaml_kwargs["task"] = "classification"
                flaml_kwargs["metric"] = "log_loss"
            else:
                flaml_kwargs["task"] = "regression"
                flaml_kwargs["metric"] = "mse"

            INFO(f"Searching for {model_name}:")
            model = self._run_automl(**flaml_kwargs)

            del flaml_kwargs["dataframe"]
            DEBUG(f"Ran AutoML with parameters: {flaml_kwargs}\n")

            setattr(self, f"_{model_name}", model)
            setattr(self, f"_{model_name}_specs", flaml_kwargs)

    @narrate(
        preamble=clg.AUTOML_CATE_PREAMBLE,
        epilogue=None,
    )
    @timer("Fitting Validation Estimators")
    def _fit_estimators(
        self,
        cate_estimators: Sequence[str],
        additional_cate_estimators: Sequence[AutoCateEstimator],
        splits: dict[str, Any],
    ) -> list[AutoCateEstimator]:
        estimators = self._get_cate_estimators(
            cate_estimators=cate_estimators,
            additional_cate_estimators=additional_cate_estimators,
        )
        if len(estimators) == 0:
            raise ValueError("No valid CATE estimators found.")

        Y_train = splits["Y_train"]
        T_train = splits["T_train"]
        X_train = splits["X_train"]
        W_train = splits["W_train"]

        def fit_estimator(estimator, Y, T, X, W) -> AutoCateEstimator | None:
            est = estimator.estimator
            est_name = estimator.name
            if isinstance(est, _OrthoLearner):
                use_W = True
            else:
                use_W = False
            if (
                isinstance(est, NonParamDML)
                and self._discrete_treatment
                and T.iloc[:, 0].nunique() > 2
            ):
                WARNING(
                    f"Non-Parametric DML models only support 1D binary or 1D continuous treatments. Skipping {est_name}."
                )
                return None
            if use_W:
                est.fit(
                    Y=Y,
                    T=T,
                    X=X if not X.empty else None,
                    W=W if not W.empty else None,
                )
            else:
                if not W.empty:
                    WARNING(
                        f"Non-Orthogonal Learners are not supported with 'W'. Skipping {est_name}."
                    )
                    return None
                else:
                    est.fit(Y=Y, T=T, X=X if not X.empty else None)
            return estimator

        if self._use_ray:
            Y_train_ref = ray.put(Y_train)
            T_train_ref = ray.put(T_train)
            X_train_ref = ray.put(X_train)
            W_train_ref = ray.put(W_train)
            remote_fns = [
                ray.remote(fit_estimator)
                .options(**self._ray_remote_func_options_kwargs)  # pyright: ignore[reportAttributeAccessIssue]
                .remote(est, Y_train_ref, T_train_ref, X_train_ref, W_train_ref)
                for est in estimators
            ]
            fitted_est = ray.get(remote_fns)
        elif self._n_jobs == 1:
            fitted_est = [
                fit_estimator(est, Y_train, T_train, X_train, W_train)
                for est in estimators
            ]
        else:
            fitted_est = Parallel(n_jobs=self._n_jobs)(
                delayed(fit_estimator)(est, Y_train, T_train, X_train, W_train)
                for est in estimators
            )

        fitted_est = [est for est in fitted_est if est is not None]

        if len(fitted_est) == 0:
            raise ValueError(
                "No valid estimators were fitted. Please check your specified CATE estimators."
            )

        return fitted_est

    @narrate(preamble=None)
    @timer("Scoring Estimators on Validation Set")
    def _validate(
        self,
        fitted_estimators: list[AutoCateEstimator],
        splits: dict[str, Any],
        ensemble: bool,
        rscorer_kwargs: dict,
    ):
        Y_val = splits["Y_val"]
        T_val = splits["T_val"]
        X_val = splits["X_val"]
        W_val = splits["W_val"]

        base_rscorer_settings = {
            "cv": 3,
            "random_state": self._seed,
        }

        if rscorer_kwargs is not None:
            base_rscorer_settings.update(rscorer_kwargs)

        rscorer = RScorer(
            model_y=self._model_Y,
            model_t=self._model_T,
            discrete_treatment=self._discrete_treatment,
            discrete_outcome=self._discrete_outcome,
            **base_rscorer_settings,
        )

        rscorer.fit(
            y=Y_val,
            T=T_val,
            X=X_val if not X_val.empty else None,
            W=W_val if not W_val.empty else None,
        )

        estimators = {mdl.name: mdl.estimator for mdl in fitted_estimators}

        best_estimator, best_score, estimator_scores = rscorer.best_model(  # pyright: ignore[reportAssignmentType]
            list(estimators.values()), return_scores=True
        )

        estimator_scores = dict(
            zip(list(estimators.keys()), estimator_scores, strict=False)
        )

        if ensemble:
            ensemble_estimator, ensemble_score, _ = rscorer.ensemble(  # pyright: ignore[reportAssignmentType]
                list(estimators.values()), return_scores=True
            )
            estimator_scores["ensemble"] = ensemble_score

            if ensemble_score > best_score:
                best_estimator = ensemble_estimator
                best_score = ensemble_score

        self.rscores = estimator_scores

        best_estimator_loc = np.argmax(list(estimator_scores.values()))
        best_estimator_name = list(estimator_scores.keys())[best_estimator_loc]

        INFO(f"Best Estimator: '{best_estimator_name}'")
        INFO(f"Estimator RScores: {estimator_scores}")

        self._best_estimator_name = best_estimator_name
        self._best_estimator = best_estimator

    @narrate(preamble=clg.CATE_TESTING_PREAMBLE)
    @timer("Verifying Final Model")
    def _test(
        self,
        splits: dict[str, Any],
        n_groups: int,
        n_bootstrap: int,
        rscorer_kwargs: dict,
    ):
        Y_train = splits["Y_train"]
        T_train = splits["T_train"]
        X_train = splits["X_train"]

        Y_test = splits["Y_test"]
        T_test = splits["T_test"]
        X_test = splits["X_test"]
        W_test = splits["W_test"]

        if not self._discrete_treatment:
            INFO("Continuous treatment specified. Using RScorer for final testing.")
            base_rscorer_settings = {
                "cv": 3,
                "random_state": self._seed,
            }

            if rscorer_kwargs is not None:
                base_rscorer_settings.update(rscorer_kwargs)

            rscorer = RScorer(
                model_y=self._model_Y,
                model_t=self._model_T,
                discrete_treatment=self._discrete_treatment,
                discrete_outcome=self._discrete_outcome,
                **base_rscorer_settings,
            )

            rscorer.fit(
                y=Y_test,
                T=T_test,
                X=X_test if not X_test.empty else None,
                W=W_test if not W_test.empty else None,
            )

            rscore = rscorer.score(self._best_estimator)

            INFO(f"RScore for {self._best_estimator_name}: {rscore}")

            self.test_results = {self._best_estimator_name: rscore}

        else:
            INFO("Discrete treatment specified. Using DRTester for final testing.")
            validator = DRTester(
                model_regression=self._model_regression,
                model_propensity=self._model_T,
                cate=self._best_estimator,
                cv=3,
            ).fit_nuisance(
                X_test.to_numpy(),
                T_test.to_numpy().ravel(),
                Y_test.to_numpy().ravel(),
                X_train.to_numpy(),
                T_train.to_numpy().ravel(),
                Y_train.to_numpy().ravel(),
            )

            res = validator.evaluate_all(
                X_test.to_numpy(),
                X_train.to_numpy(),
                n_groups=n_groups,
                n_bootstrap=n_bootstrap,
            )

            summary = res.summary()
            if np.array(
                summary[[c for c in summary.columns if "pval" in c]] > 0.1
            ).any():
                WARNING(
                    "Some of the validation results suggest that the model may not have found statistically significant heterogeneity. Please closely look at the validation results and consider retraining with new configurations.\n"
                )
            else:
                INFO(
                    "All validation results suggest that the model has found statistically significant heterogeneity.\n"
                )

            INFO(summary)
            for i in res.blp.treatments:
                if i > 0:
                    INFO("CALIBRATION CURVE")
                    res.plot_cal(i)
                    plt.show()
                    INFO("QINI CURVE")
                    res.plot_qini(i)
                    plt.show()
                    INFO("TOC CURVE")
                    res.plot_toc(i)
                    plt.show()

            self.test_results = res

    @narrate(preamble=clg.REFIT_FINAL_PREAMBLE)
    @timer("Refitting Final Estimator")
    def refit_final(self, df: PandasConvertibleDataFrame):
        pd_df = self._convert_dataframe_to_pandas(df)
        if self._enable_categorical:
            pd_df, _ = self._encode_categoricals(
                pd_df, categorical_mappings=self._categorical_mappings
            )
        estimator = self._best_estimator
        Y = pd_df[self.Y]
        T = pd_df[self.T]
        X = pd_df[self.X]
        W = pd_df[self.W]

        def fit_model(est):
            if isinstance(est, _OrthoLearner):
                est.fit(
                    Y=Y, T=T, X=X if not X.empty else None, W=W if not W.empty else None
                )
            else:
                est.fit(Y=Y, T=T, X=X if not X.empty else None)

        if isinstance(estimator, EnsembleCateEstimator):
            for est in estimator._cate_models:
                fit_model(est)
        else:
            fit_model(estimator)

    def _get_cate_estimators(
        self,
        *,
        cate_estimators: Sequence[str],
        additional_cate_estimators: Sequence[AutoCateEstimator],
    ) -> list[AutoCateEstimator]:
        estimators: list[AutoCateEstimator] = []
        for est in cate_estimators:
            estimators.append(available_estimators[est])

        additional_cate_estimators = list(additional_cate_estimators)
        estimators = estimators + additional_cate_estimators

        to_remove = []
        for ace in estimators:
            name = ace.name
            estimator = ace.estimator

            def check_for_instance(estimator, models):
                for model in models:
                    try:
                        inst = getattr(dml, model)
                    except AttributeError:
                        try:
                            inst = getattr(dr, model)
                        except AttributeError:
                            inst = getattr(metalearners, model)

                    if type(estimator) is inst:
                        return True
                return False

            if self._discrete_outcome:
                res = check_for_instance(
                    estimator,
                    ["NonParamDML", "SLearner", "TLearner", "XLearner", "DRLearner"],
                )
                if res:
                    WARNING(
                        f"Discrete outcomes not yet supported for {name}! Removing..."
                    )
                    to_remove.append(ace)

            if not self._discrete_treatment:
                res = check_for_instance(
                    estimator,
                    [
                        "DRLearner",
                        "ForestDRLearner",
                        "LinearDRLearner",
                        "SLearner",
                        "TLearner",
                        "XLearner",
                    ],
                )
                if res:
                    WARNING(
                        f"Discrete treatments not yet supported for {name}! Removing..."
                    )
                    to_remove.append(ace)

            attr_mapping = {
                "model_y": self._model_Y,
                "model_t": self._model_T,
                "model_regression": self._model_regression,
                "model_final": self._model_regression,
                "propensity_model": self._model_T,
                "model_propensity": self._model_T,
                "overall_model": self._model_regression,
                "models": self._model_regression,
                "cate_models": self._model_regression,
            }

            for k, v in attr_mapping.items():
                if hasattr(estimator, k):
                    if getattr(estimator, k) == "auto":
                        try:
                            setattr(estimator, k, v)
                        except ValueError:
                            pass

            if hasattr(estimator, "discrete_outcome"):
                setattr(estimator, "discrete_outcome", self._discrete_outcome)
            if hasattr(estimator, "discrete_treatment"):
                setattr(estimator, "discrete_treatment", self._discrete_treatment)
            if hasattr(estimator, "random_state"):
                setattr(estimator, "random_state", self._seed)

        for est in to_remove:
            if est in estimators:
                estimators.remove(est)
        return estimators

    def __str__(self):
        """
        Returns a string representation of the AutoCATE object.

        Returns
        -------
        str
            A string containing information about the AutoCATE object, including data backend, number of observations, UUID, outcome variable, discrete outcome, treatment variable, discrete treatment, features/confounders, random seed, nuissance models (if fitted), and final estimator (if available).
        """
        summary = (
            "================== AutoCATE Object ==================\n"
            + f"Outcome Variable: {self.Y}\n"
            + f"Discrete Outcome: {self._discrete_outcome}\n"
            + f"Treatment Variable: {self.T}\n"
            + f"Discrete Treatment: {self._discrete_treatment}\n"
            + f"Features/Confounders for Heterogeneity (X): {self.X}\n"
            + f"Features/Confounders as Controls (W): {self.W}\n"
            + f"Enable Categorical: {self._enable_categorical}\n"
            + f"n Jobs: {self._n_jobs}\n"
            + f"Use Ray: {self._use_ray}\n"
            + f"Use Spark: {self._use_spark}\n"
            + f"Random Seed: {self._seed}\n"
        )

        if self._fitted:
            summary += (
                f"Nuisance Model Y: {self._model_Y}\n"
                + f"Propensity/Nuisance Model T: {self._model_T}\n"
                + f"Regression Model: {self._model_regression}\n"
                + f"Best Estimator: {self._best_estimator_name}\n"
            )

        return summary
