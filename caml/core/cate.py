from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml._cate_estimator import BaseCateEstimator
from econml._ortho_learner import _OrthoLearner
from econml.dml.dml import NonParamDML
from econml.score import EnsembleCateEstimator
from econml.validate.drtester import DRTester
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

from ..generics import (
    FittedAttr,
    PandasConvertibleDataFrame,
    experimental,
    is_module_available,
    timer,
)
from ..logging import ERROR, INFO, WARNING
from ..monkey_patch import RScorer
from ._base import BaseCamlEstimator
from .modeling.model_bank import (
    AutoCateEstimator,
    available_estimators,
    get_cate_estimator,
)

_HAS_PYSPARK = is_module_available("pyspark")
_HAS_RAY = is_module_available("ray")

if _HAS_RAY:
    import ray

if TYPE_CHECKING:
    import ray


warnings.filterwarnings(
    "ignore", message="A column-vector y was passed when a 1d array was expected"
)


# Your code here
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

    validation_estimator = FittedAttr("_validation_estimator")
    final_estimator = FittedAttr("_final_estimator")
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

    def fit(
        self,
        df: PandasConvertibleDataFrame,
        cate_estimators: Sequence[str] = available_estimators,
        additional_cate_estimators: Sequence[AutoCateEstimator] = [],
        ensemble: bool = False,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.1,
    ):
        if self.use_ray:
            if not ray.is_initialized():
                ray.init()
        ## Add argument checks (e.g. validate cate_estimators and additional_cate_estimators)
        pd_df = self._convert_dataframe_to_pandas(df)
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
        final_estimator = self._validate(fitted_estimators, splits)
        # self._test(final_estimator, splits)
        return final_estimator

    def estimate_ate(self, df: PandasConvertibleDataFrame) -> None:
        return

    def estimate_cate(self, df: PandasConvertibleDataFrame) -> None:
        return

    def predict(self, df: PandasConvertibleDataFrame) -> None:
        return

    @timer("Find Nuisance Functions")
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
                self.X + self.W + self.T,
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

            model = self._run_automl(**flaml_kwargs)

            setattr(self, model_name, model)
            setattr(self, f"_{model_name}", flaml_kwargs)

        self._nuisances_fitted = True

    @timer("Fit Validation Estimators")
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
                    est.fit(Y=Y, T=T, X=X)
            return estimator

        if self.use_ray:
            Y_train_ref = ray.put(Y_train)
            T_train_ref = ray.put(T_train)
            X_train_ref = ray.put(X_train)
            W_train_ref = ray.put(W_train)
            remote_fns = [
                ray.remote(fit_estimator)
                .options(**self.ray_remote_func_options_kwargs)  # type: ignore
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

    @timer("Score Estimators on Validation Set")
    def _validate(
        self, fitted_estimators: list[AutoCateEstimator], splits: dict[str, Any]
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
            **base_rscorer_settings,
        )

        rscorer.fit(
            y=Y_val,
            T=T_val,
            X=X_val if not X_val.empty else None,
            W=W_val if not W_val.empty else None,
            discrete_outcome=self.discrete_outcome,  # type: ignore
        )

        best_estimator, _, estimator_scores = rscorer.best_model(  # type: ignore
            [mdl.estimator for mdl in fitted_estimators], return_scores=True
        )

        estimator_scores = dict(
            zip([mdl.name for mdl in fitted_estimators], estimator_scores, strict=False)
        )

        INFO(f"Best Estimator: {best_estimator}")
        INFO(f"Estimator RScores: {estimator_scores}")
        print(f"Best Estimator: {best_estimator}")
        print(f"Estimator RScores: {estimator_scores}")

        self.rscores = estimator_scores

        return best_estimator

        # if ensemble:
        #     ensemble_estimator, ensemble_score, estimator_scores = rscorer.ensemble(
        #         [mdl for _, mdl in models], return_scores=True
        #     )
        #     estimator_scores = list(estimator_scores)
        #     estimator_scores.append(ensemble_score)
        #     models.append(("ensemble", ensemble_estimator))
        #     self.cate_estimators.append(("ensemble", ensemble_estimator))

    def validate(
        self,
        *,
        n_groups: int = 4,
        n_bootstrap: int = 100,
        estimator: BaseCateEstimator | EnsembleCateEstimator | None = None,
        print_full_report: bool = True,
    ):
        """
        Validates the fitted CATE models on the test set to check for generalization performance.

        Uses the DRTester class from EconML to obtain the Best Linear Predictor (BLP), Calibration, AUTOC, and QINI.
        See [EconML documentation](https://econml.azurewebsites.net/_autosummary/econml.validate.DRTester.html) for more details.
        In short, we are checking for the ability of the model to find statistically significant heterogeneity in a "well-calibrated" fashion.

        Sets the `validator_report` attribute to the validation report.

        Parameters
        ----------
        n_groups : int
            The number of quantile based groups used to calculate calibration scores.
        n_bootstrap : int
            The number of boostrap samples to run when calculating confidence bands.
        estimator : BaseCateEstimator | EnsembleCateEstimator | None
            The estimator to validate. Default implies the best estimator from the validation set.
        print_full_report : bool
            A boolean indicating whether to print the full validation report.

        Examples
        --------
        ```{python}
        caml_obj.validate()

        caml_obj.validator_results
        ```
        """
        plt.style.use("ggplot")

        if estimator is None:
            estimator = self._validation_estimator

        if not self.discrete_treatment or self.discrete_outcome:
            ERROR(
                "Validation for continuous treatments and/or discrete outcomes is not supported yet."
            )
            return

        validator = DRTester(
            model_regression=self.model_Y_X_W_T,
            model_propensity=self.model_T_X_W,
            cate=estimator,
            cv=3,
        )

        X_test, W_test, T_test, Y_test = (
            self._data_splits["X_test"],
            self._data_splits["W_test"],
            self._data_splits["T_test"],
            self._data_splits["Y_test"],
        )

        X_train, W_train, T_train, Y_train = (
            self._data_splits["X_train"],
            self._data_splits["W_train"],
            self._data_splits["T_train"],
            self._data_splits["Y_train"],
        )

        X_W_test = np.hstack((X_test, W_test))
        X_W_train = np.hstack((X_train, W_train))

        validator.fit_nuisance(
            X_W_test,
            T_test.astype(int),
            Y_test,
            X_W_train,
            T_train.astype(int),
            Y_train,
        )

        res = validator.evaluate_all(
            X_test, X_train, n_groups=n_groups, n_bootstrap=n_bootstrap
        )

        # Check for insignificant results & warn user
        summary = res.summary()
        if np.array(summary[[c for c in summary.columns if "pval" in c]] > 0.1).any():
            WARNING(
                "Some of the validation results suggest that the model may not have found statistically significant heterogeneity. Please closely look at the validation results and consider retraining with new configurations."
            )
        else:
            INFO(
                "All validation results suggest that the model has found statistically significant heterogeneity."
            )

        if print_full_report:
            print(summary.to_string())
            for i in res.blp.treatments:
                if i > 0:
                    res.plot_cal(i)
                    res.plot_qini(i)
                    res.plot_toc(i)

        self.validator_results = res

    def fit_final(self):
        """
        Fits the final estimator on the entire dataset, after validation and testing.

        Sets the `input_names` and `final_estimator` class attributes.

        Examples
        --------
        ```{python}
        caml_obj.fit_final()

        print(caml_obj.final_estimator)
        print(caml_obj.input_names)
        ```
        """
        self.input_names = {}
        if not self._validation_estimator:
            raise RuntimeError(
                "Must fit validation estimator first before fitting final estimator. Please run fit_validator() method first."
            )
        self._final_estimator = copy.deepcopy(self._validation_estimator)

        Y, T, X, W = self._Y, self._T, self._X, self._W

        if isinstance(self._final_estimator, EnsembleCateEstimator):
            for estimator in self._final_estimator._cate_models:
                if isinstance(estimator, _OrthoLearner):
                    estimator.fit(
                        Y=Y,
                        T=T,
                        X=X,
                        W=W if W.shape[1] > 0 else None,
                    )
                else:
                    estimator.fit(
                        Y=Y,
                        T=T,
                        X=X,
                    )
                    self.input_names["feature_names"] = self.X
                    self.input_names["output_names"] = self.Y
                    self.input_names["treatment_names"] = self.T
        else:
            if isinstance(self._final_estimator, _OrthoLearner):
                self._final_estimator.fit(
                    Y=Y,
                    T=T,
                    X=X,
                    W=W if W.shape[1] > 0 else None,
                )
            else:
                self._final_estimator.fit(
                    Y=Y,
                    T=T,
                    X=X,
                )

            self.input_names["feature_names"] = self.X
            self.input_names["output_names"] = self.Y
            self.input_names["treatment_names"] = self.T

    def predict_old(
        self,
        *,
        X: pd.DataFrame | np.ndarray | None = None,
        T0: int = 0,
        T1: int = 1,
        T: pd.DataFrame | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predicts the CATE based on the fitted final estimator for either the internal dataset or provided Data.

        For binary treatments, the CATE is the estimated effect of the treatment and for a continuous treatment, the CATE is the estimated effect of a one-unit increase in the treatment.
        This can be modified by setting the T0 and T1 parameters to the desired treatment levels.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray | None
            The DataFrame containing the features (X) for which CATE needs to be predicted.
            If not provided, defaults to the internal dataset.
        T0 : int
            Base treatment for each sample.
        T1 : int
            Target treatment for each sample.
        T : pd.DataFrame | np.ndarray | None
            Treatment vector if continuous treatment is leveraged for computing marginal effects around treatments for each individual.

        Returns
        -------
        np.ndarray
            The predicted CATE values if return_predictions is set to True.

        Examples
        --------
        ```{python}
        caml_obj.predict()
        ```
        """
        if not self._final_estimator:
            raise RuntimeError(
                "Must fit final estimator first before making predictions. Please run fit_final() method first."
            )

        if X is None:
            _X = self._X
            _T = self._T
        else:
            _X = X
            _T = T

        if self.discrete_treatment:
            cate_predictions = self._final_estimator.effect(_X, T0=T0, T1=T1)
        else:
            cate_predictions = self._final_estimator.marginal_effect(_T, _X)

        if cate_predictions.ndim > 1:
            cate_predictions = cate_predictions.ravel()

        if X is None:
            self._cate_predictions[f"cate_predictions_{T0}_{T1}"] = cate_predictions

        return cate_predictions

    def summarize(
        self,
        *,
        cate_predictions: np.ndarray | None = None,
    ):
        """
        Provides population summary statistics for the CATE predictions for either the internal results or provided results.

        Parameters
        ----------
        cate_predictions : np.ndarray | None
            The CATE predictions for which summary statistics will be generated.
            If not provided, defaults to internal CATE predictions generated by `predict()` method with X=None.

        Returns
        -------
        pd.DataFrame | pd.Series
            The summary statistics for the CATE predictions.

        Examples
        --------
        ```{python}
        caml_obj.summarize()
        ```
        """
        if cate_predictions is None:
            _cate_predictions = self._cate_predictions
            cate_predictions_df = pd.DataFrame.from_dict(_cate_predictions)
        else:
            _cate_predictions = cate_predictions
            cate_predictions_df = pd.DataFrame(
                cate_predictions, columns=["cate_predictions"]
            )

        return cate_predictions_df.describe()

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
                f"Nuissance Model Y_X: {self.model_Y}\n"
                + f"Propensity/Nuissance Model T_X: {self.model_T}\n"
                + f"Regression Model Y_X_T: {self.model_regression}\n"
            )

        if self._final_estimator is not None:
            summary += f"Final Estimator: {self._final_estimator}\n"

        return summary
