from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml._cate_estimator import BaseCateEstimator
from econml._ortho_learner import _OrthoLearner
from econml.dml.dml import NonParamDML
from econml.score import EnsembleCateEstimator, RScorer
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

from caml import logging as clg
from caml.core._base import BaseCamlEstimator
from caml.core.modeling.model_bank import (
    AutoCateEstimator,
    available_estimators,
    get_cate_estimator,
)
from caml.generics.decorators import experimental, narrate, timer
from caml.generics.interfaces import FittedAttr, PandasConvertibleDataFrame
from caml.generics.monkey_patch import DRTester
from caml.generics.utils import is_module_available
from caml.logging import INFO, WARNING

_HAS_PYSPARK = is_module_available("pyspark")
_HAS_RAY = is_module_available("ray")

if _HAS_RAY:
    import ray

if TYPE_CHECKING:
    import ray


warnings.filterwarnings("ignore")


# TODO: Refactor all docstrings!!
@experimental
class AutoCATE(BaseCamlEstimator):
    r"""The AutoCATE class represents an opinionated framework of Causal Machine Learning techniques for estimating highly accurate conditional average treatment effects (CATEs).

    **AutoCATE is experimental and may change significantly in future versions.**

    The CATE is defined formally as $\mathbb{E}[\tau|\mathbf{X}]$
    where $\tau$ is the treatment effect and $\mathbf{X}$ is the set of covariates.

    This class is built on top of the EconML library and provides a high-level API for fitting, validating, and making inference with CATE models,
    with best practices built directly into the API. The class is designed to be easy to use and understand, while still providing
    flexibility for advanced users. The class is designed to be used with `pandas`, `polars`, or `pyspark` backends, which ultimately get
    converted to NumPy Arrays under the hood to provide a level of extensibility & interoperability across different data processing frameworks.

    The primary workflow for the AutoCATE class is as follows:

    1. Initialize the class with the input DataFrame and the necessary columns.
    2. Utilize [flaml](https://microsoft.github.io/FLAML/) AutoML to find nuisance functions or propensity/regression models to be utilized in the EconML estimators.
    3. Fit the CATE models on the training set and select top performer based on the RScore from validation set.
    4. Validate the fitted CATE model on the test set to check for generalization performance.
    5. Fit the final estimator on the entire dataset, after validation and testing.
    6. Predict the CATE based on the fitted final estimator for either the internal dataset or an out-of-sample dataset.
    8. Summarize population summary statistics for the CATE predictions for either the internal dataset or out-of-sample predictions.

    For technical details on conditional average treatment effects, see:

     - CaML Documentation
     - [EconML documentation](https://econml.azurewebsites.net/)

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
    X : str | Sequence[str]
        The str (if unity) or Sequence of feature names representing the feature set to be utilized for estimating heterogeneity/CATE.
    W : str | Sequence[str] | None
        The str (if unity) or Sequence of feature names representing the confounder/control feature set to be utilized only for nuisance function estimation. When W is passed, only Orthogonal learners will be leveraged.
    discrete_treatment : bool
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome : bool
        A boolean indicating whether the outcome is binary or continuous.
    seed : int | None
        The seed to use for the random number generator.

    Attributes
    ----------
    Y : str
        The str representing the column name for the outcome variable.
    T : str
        The str representing the column name(s) for the treatment variable(s).
    X : Sequence[str]
        The str (if unity) or Sequence of variable names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATE and nuisance function estimation where applicable.
    W : Sequence[str] | None
        The str (if unity) or Sequence of variable names representing the confounder/control feature set to be utilized only for nuisance function estimation, where applicable. These will be included by default in Meta-Learners.
    discrete_treatment : bool
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome : bool
        A boolean indicating whether the outcome is binary or continuous.
    available_estimators : list[str]
        A list of the available CATE estimators out of the box. Validity of estimator at runtime will depend on the outcome and treatment types and be automatically selected.
    model_Y_X_W: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable.
    model_Y_X_W_T: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable with treatment variable.
    model_T_X_W: sklearn.base.BaseEstimator
        The fitted nuisance function for the treatment variable.
    cate_estimators: dict[str, AutoCateEstimator]
        Dictionary of fitted cate estimator objects.
    rscores: dict[str, float]
        Dictionary of RScore values for each fitted cate estimator.
    validation_estimator : AutoCateEstimator
        The fitted EconML estimator object for validation.
    validator_results : econml.validate.results.EvaluationResults
        The validation results object.
    final_estimator : AutoCateEstimator
        The fitted EconML estimator object on the entire dataset after validation.
    input_names : dict[str,list[str]]
        The feature, outcome, and treatment names used in the CATE estimators.

    Examples
    --------
    ```{python}
    from caml import AutoCATE
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data_generator = SyntheticDataGenerator(seed=10, n_cont_modifiers=1, n_cont_confounders=1)
    df = data_generator.df

    caml_obj = AutoCATE(
        df = df,
        Y="Y1_continuous",
        T="T1_binary",
        X=[c for c in df.columns if "X" in c or "W" in c],
        discrete_treatment=True,
        discrete_outcome=False,
        seed=0,
    )

    print(caml_obj)
    ```
    """

    best_estimator: BaseCateEstimator | EnsembleCateEstimator = FittedAttr(
        "_best_estimator"
    )  # pyright: ignore[reportAssignmentType]
    model_Y: BaseEstimator
    model_T: BaseEstimator
    model_regression: BaseEstimator

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
        n_jobs: int = 1,
        use_ray: bool = False,
        ray_remote_func_options_kwargs: dict | None = None,
        use_spark: bool = False,
        seed: int | None = None,
    ):
        self.Y = [Y] if isinstance(Y, str) else list(Y)
        self.T = [T] if isinstance(T, str) else list(T)
        self.X = list(X) if X else list()
        self.W = list(W) if W else list()
        self.discrete_treatment = discrete_treatment
        self.discrete_outcome = discrete_outcome
        self._model_Y = model_Y
        self._model_T = model_T
        self._model_regression = model_regression
        self.n_jobs = n_jobs
        self.use_ray = use_ray
        self.ray_remote_func_options_kwargs = (
            ray_remote_func_options_kwargs
            if ray_remote_func_options_kwargs is not None
            else {}
        )
        self.use_spark = use_spark
        self.seed = seed
        self.available_estimators = available_estimators

        self._fitted = False
        self._nuisances_fitted = False
        self._cate_predictions = {}

        if not discrete_treatment:
            WARNING("Validation for continuous treatments is not supported yet.")

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
        cate_estimators: Sequence[str] = available_estimators,
        additional_cate_estimators: Sequence[AutoCateEstimator] = list(),
        ensemble: bool = False,
        refit_final: bool = True,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.1,
    ):
        """TODO: Docstring."""
        INFO(f"{self} \n")
        if self.use_ray:
            if not ray.is_initialized():
                ray.init()
        ## Add argument checks (e.g. validate cate_estimators and additional_cate_estimators)
        pd_df = self._convert_dataframe_to_pandas(df, encode_categoricals=True)
        self._find_nuisance_functions(pd_df)
        splits = self._split_data(
            df=pd_df,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        estimators = self._get_cate_estimators(
            cate_estimators=list(cate_estimators),
            additional_cate_estimators=list(additional_cate_estimators),
        )
        fitted_estimators = self._fit_estimators(estimators, splits)
        self._validate(fitted_estimators, splits, ensemble)
        self._test(splits, n_groups=5, n_bootstrap=100)

        self.fitted = True

        if refit_final:
            self.refit_final(pd_df)

    @timer("Estimate ATEs")
    def estimate_ate(
        self, df: PandasConvertibleDataFrame, T0: int = 0, T1: int = 1
    ) -> float:
        """TODO: Docstring."""
        cates = self.estimate_cate(df, T0=T0, T1=T1)
        ate = np.mean(cates)

        return ate  # pyright: ignore[reportReturnType]

    @timer("Estimating CATEs")
    def estimate_cate(
        self, df: PandasConvertibleDataFrame, T0: int = 0, T1: int = 1
    ) -> np.ndarray:
        """TODO: Docstring."""
        pd_df = self._convert_dataframe_to_pandas(df, encode_categoricals=True)
        if self.discrete_treatment:
            cates = self.best_estimator.effect(pd_df[self.X], T0=T0, T1=T1)
        else:
            cates = self.best_estimator.marginal_effect(pd_df[self.T], pd_df[self.X])

        return cates

    def predict(self, df: PandasConvertibleDataFrame, **kwargs) -> np.ndarray:
        """Alias for `estimate_cate`."""
        return self.estimate_cate(df, **kwargs)

    def effect(self, df: PandasConvertibleDataFrame, **kwargs) -> np.ndarray:
        """Alias for `estimate_cate`."""
        return self.estimate_cate(df, **kwargs)

    @narrate(preamble=clg.AUTOML_NUISANCE_PREAMBLE)
    @timer("Nuisance Function AutoML")
    def _find_nuisance_functions(self, df: pd.DataFrame):
        base_settings = {
            "n_jobs": -1,
            "log_file_name": "",
            "seed": self.seed,
            "time_budget": 300,
            "early_stop": "True",
            "eval_method": "cv",
            "n_splits": 3,
            "starting_points": "static",
            "estimator_list": "auto",
            "verbose": 0,
        }

        if self.use_spark:
            base_settings["use_spark"] = True
            base_settings["n_concurrent_trials"] = 4
        elif self.use_ray:
            base_settings["use_ray"] = True
            base_settings["n_concurrent_trials"] = 4

        # Model configurations: (model_name, outcome, features, discrete_outcome)
        model_configs = [
            ("model_Y", self.Y, self.X + self.W, self.discrete_outcome),
            (
                "model_regression",
                self.Y,
                self.X + self.W + list(self.T),
                self.discrete_outcome,
            ),
            ("model_T", self.T, self.X + self.W, self.discrete_treatment),
        ]

        for model_name, outcome, features, discrete_outcome in model_configs:
            flaml_kwargs = base_settings.copy()

            model_arg = getattr(self, f"_{model_name}")

            if isinstance(model_arg, dict):
                flaml_kwargs.update(model_arg)
            elif model_arg is None:
                pass
            else:
                setattr(self, model_name, model_arg)
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

            setattr(self, model_name, model)
            setattr(self, f"_{model_name}", flaml_kwargs)

        self._nuisances_fitted = True

    @narrate(
        preamble=clg.AUTOML_CATE_PREAMBLE,
        epilogue=None,
    )
    @timer("Fitting Validation Estimators")
    def _fit_estimators(
        self, cate_estimators: list[AutoCateEstimator], splits: dict[str, Any]
    ) -> list[AutoCateEstimator]:
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
                and self.discrete_treatment
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

        if self.use_ray:
            Y_train_ref = ray.put(Y_train)
            T_train_ref = ray.put(T_train)
            X_train_ref = ray.put(X_train)
            W_train_ref = ray.put(W_train)
            remote_fns = [
                ray.remote(fit_estimator)
                .options(**self.ray_remote_func_options_kwargs)  # pyright: ignore[reportAttributeAccessIssue]
                .remote(est, Y_train_ref, T_train_ref, X_train_ref, W_train_ref)
                for est in cate_estimators
            ]
            fitted_est = ray.get(remote_fns)
        elif self.n_jobs == 1:
            fitted_est = [
                fit_estimator(est, Y_train, T_train, X_train, W_train)
                for est in cate_estimators
            ]
        else:
            fitted_est = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_estimator)(est, Y_train, T_train, X_train, W_train)
                for est in cate_estimators
            )

        fitted_est = [est for est in fitted_est if est is not None]

        return fitted_est

    @narrate(preamble=None)
    @timer("Scoring Estimators on Validation Set")
    def _validate(
        self,
        fitted_estimators: list[AutoCateEstimator],
        splits: dict[str, Any],
        ensemble: bool = True,
    ):
        Y_val = splits["Y_val"]
        T_val = splits["T_val"]
        X_val = splits["X_val"]
        W_val = splits["W_val"]

        base_rscorer_settings = {
            "cv": 3,
            "mc_iters": 3,
            "mc_agg": "median",
            "random_state": self.seed,
        }

        # if rscorer_kwargs is not None:
        #     base_rscorer_settings.update(rscorer_kwargs)

        rscorer = RScorer(
            model_y=self.model_Y,
            model_t=self.model_T,
            discrete_treatment=self.discrete_treatment,
            discrete_outcome=self.discrete_outcome,
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

        self.best_estimator_name = best_estimator_name
        self.best_estimator = best_estimator

    @narrate(preamble=clg.CATE_TESTING_PREAMBLE)
    @timer("Verifying Final Model")
    def _test(
        self,
        splits: dict[str, Any],
        n_groups: int,
        n_bootstrap: int,
    ):
        Y_train = splits["Y_train"]
        T_train = splits["T_train"]
        X_train = splits["X_train"]
        W_train = splits["W_train"]

        Y_test = splits["Y_test"]
        T_test = splits["T_test"]
        X_test = splits["X_test"]
        W_test = splits["W_test"]

        X_W_test = pd.concat((X_test, W_test), axis=1)
        X_W_train = pd.concat((X_train, W_train), axis=1)

        if not self.discrete_treatment:
            INFO("Continuous treatment specified. Using RScorer for final testing.")
            base_rscorer_settings = {
                "cv": 3,
                "mc_iters": 3,
                "mc_agg": "median",
                "random_state": self.seed,
            }

            # if rscorer_kwargs is not None:
            #     base_rscorer_settings.update(rscorer_kwargs)

            rscorer = RScorer(
                model_y=self.model_Y,
                model_t=self.model_T,
                discrete_treatment=self.discrete_treatment,
                discrete_outcome=self.discrete_outcome,
                **base_rscorer_settings,
            )

            rscorer.fit(
                y=Y_test,
                T=T_test,
                X=X_test if not X_test.empty else None,
                W=W_test if not W_test.empty else None,
            )

            rscore = rscorer.score(self.best_estimator)

            INFO(f"RScore for {self.best_estimator_name}: {rscore}")

            self.test_results = {self.best_estimator_name: rscore}

        else:
            INFO("Discrete treatment specified. Using DRTester for final testing.")
            validator = DRTester(
                model_regression=self.model_regression,
                model_propensity=self.model_T,
                cate=self.best_estimator,
                cv=3,
            ).fit_nuisance(
                X_W_test.to_numpy(),
                T_test.to_numpy().ravel(),
                Y_test.to_numpy().ravel(),
                X_W_train.to_numpy(),
                T_train.to_numpy().ravel(),
                Y_train.to_numpy().ravel(),
            )

            res = validator.evaluate_all(
                X_W_test.to_numpy(),
                X_W_train.to_numpy(),
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
        df = self._convert_dataframe_to_pandas(df, encode_categoricals=True)
        estimator = self.best_estimator
        Y = df[self.Y]
        T = df[self.T]
        X = df[self.X]
        W = df[self.W]

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
        cate_estimators: list[str],
        additional_cate_estimators: list[AutoCateEstimator],
    ) -> list[AutoCateEstimator]:
        _cate_estimators: list[AutoCateEstimator] = []
        for est in cate_estimators:
            estimator = get_cate_estimator(
                est,
                self.model_Y,
                self.model_T,
                self.model_regression,
                self.discrete_treatment,
                self.discrete_outcome,
                self.seed,
            )
            if estimator is None:
                pass
            else:
                _cate_estimators.append(estimator)

        return _cate_estimators + additional_cate_estimators

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
            + f"Discrete Outcome: {self.discrete_outcome}\n"
            + f"Treatment Variable: {self.T}\n"
            + f"Discrete Treatment: {self.discrete_treatment}\n"
            + f"Features/Confounders for Heterogeneity (X): {self.X}\n"
            + f"Features/Confounders as Controls (W): {self.W}\n"
            + f"Random Seed: {self.seed}\n"
        )

        if self._nuisances_fitted:
            summary += (
                f"Nuissance Model Y: {self.model_Y}\n"
                + f"Propensity/Nuissance Model T: {self.model_T}\n"
                + f"Regression Model: {self.model_regression}\n"
            )

        if self._fitted:
            summary += f"Best Estimator: {self.best_estimator}\n"

        return summary
