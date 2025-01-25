from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas
from econml._cate_estimator import BaseCateEstimator
from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML
from econml.score import EnsembleCateEstimator, RScorer
from econml.validate.drtester import DRTester
from joblib import Parallel, delayed

from ..generics import cls_typechecked
from ._base import CamlBase
from .modeling import model_bank

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import pyspark

    _HAS_PYSPARK = True
except ImportError:
    _HAS_PYSPARK = False

try:
    import ray

    _HAS_RAY = True
except ImportError:
    _HAS_RAY = False

if TYPE_CHECKING:
    import polars
    import pyspark
    import ray


@cls_typechecked
class CamlCATE(CamlBase):
    r"""
    The CamlCATE class represents an opinionated framework of Causal Machine Learning techniques for estimating highly accurate conditional average treatment effects (CATEs).

    The CATE is defined formally as $\mathbb{E}[\tau|\mathbf{X}]$
    where $\tau$ is the treatment effect and $\mathbf{X}$ is the set of covariates.

    This class is built on top of the EconML library and provides a high-level API for fitting, validating, and making inference with CATE models,
    with best practices built directly into the API. The class is designed to be easy to use and understand, while still providing
    flexibility for advanced users. The class is designed to be used with `pandas`, `polars`, or `pyspark` backends, which ultimately get
    converted to NumPy Arrays under the hood to provide a level of extensibility & interoperability across different data processing frameworks.

    The primary workflow for the CamlCATE class is as follows:

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

    **Outcome/Treatment Type Support Matrix**
    <center>
    | Outcome     | Treatment   | Support     | Missing    |
    | ----------- | ----------- | ----------- | ---------- |
    | Continuous  | Binary      | ✅Full      |            |
    | Continuous  | Continuous  | 🟡Partial   | validate() |
    | Continuous  | Categorical | ✅Full      |            |
    | Binary      | Binary      | 🟡Partial   | validate() |
    | Binary      | Continuous  | 🟡Partial   | validate() |
    | Binary      | Categorical | 🟡Partial   | validate() |
    | Categorical | Binary      | ❌Not yet   |            |
    | Categorical | Continuous  | ❌Not yet   |            |
    | Categorical | Categorical | ❌Not yet   |            |

    </center>
    Multi-dimensional outcomes and treatments are not yet supported.

    Parameters
    ----------
    df
        The input DataFrame representing the data for the CamlCATE instance.
    Y
        The str representing the column name for the outcome variable.
    T
        The str representing the column name(s) for the treatment variable(s).
    X
        The str (if unity) or list of feature names representing the feature set to be utilized for estimating heterogeneity/CATE.
    W
        The str (if unity) or list of feature names representing the confounder/control feature set to be utilized only for nuisance function estimation. When W is passed, only Orthogonal learners will be leveraged.
    discrete_treatment
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome
        A boolean indicating whether the outcome is binary or continuous.
    seed
        The seed to use for the random number generator.
    verbose
        The verbosity level for logging. Default implies 1 (INFO). Set to 0 for no logging. Set to 2 for DEBUG.

    Attributes
    ----------
    df : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame
        The input DataFrame representing the data for the CamlCATE instance.
    Y : str
        The str representing the column name for the outcome variable.
    T : str
        The str representing the column name(s) for the treatment variable(s).
    X : list[str]
        The str (if unity) or list/tuple of feature names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATE and nuisance function estimation where applicable.
    W : list[str]
        The str (if unity) or list/tuple of feature names representing the confounder/control feature set to be utilized only for nuisance function estimation, where applicable. These will be included by default in Meta-Learners.
    discrete_treatment : bool
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome : bool
        A boolean indicating whether the outcome is binary or continuous.
    available_estimators : str
        A list of the available CATE estimators out of the box. Validity of estimator at runtime will depend on the outcome and treatment types and be automatically selected.
    model_Y_X_W: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable.
    model_Y_X_W_T: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable with treatment variable.
    model_T_X_W: sklearn.base.BaseEstimator
        The fitted nuisance function for the treatment variable.
    cate_estimators: dict[str, econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator]
        Dictionary of fitted cate estimator objects.
    validation_estimator : econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object for validation.
    validator_results : econml.validate.results.EvaluationResults
        The validation results object.
    final_estimator : econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object on the entire dataset after validation.
    input_names : dict[str,list[str]]
        The feature, outcome, and treatment names used in the CATE estimators.

    Examples
    --------
    ```{python}
    from caml import CamlCATE
    from caml.extensions.synthetic_data import CamlSyntheticDataGenerator

    data_generator = CamlSyntheticDataGenerator(seed=10)
    df = data_generator.df

    caml_obj = CamlCATE(
        df = df,
        Y="Y1_continuous",
        T="T1_binary",
        X=[c for c in df.columns if "X" in c or "W" in c],
        discrete_treatment=True,
        discrete_outcome=False,
        seed=0,
        verbose=1,
    )

    print(caml_obj)
    ```
    """

    def __init__(
        self,
        df: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame,
        Y: str,
        T: str,
        X: list[str],
        W: list[str] | None = None,
        *,
        discrete_treatment: bool = True,
        discrete_outcome: bool = False,
        seed: int | None = None,
        verbose: int = 1,
    ):
        self.df = df
        super().__init__(verbose=verbose)

        self.Y = Y
        self.T = T
        self.X = X
        self.W = W if W else []
        self.discrete_treatment = discrete_treatment
        self.discrete_outcome = discrete_outcome
        self.seed = seed
        self.available_estimators = model_bank.valid_models

        self._Y, self._T, self._X, self._W = self._dataframe_to_numpy()

        self._nuisances_fitted = False
        self._validation_estimator = None
        self._final_estimator = None
        self._cate_predictions = {}

        if not self.discrete_treatment:
            logger.warning("Validation for continuous treatments is not supported yet.")

        if self.discrete_outcome:
            logger.warning("Binary outcomes are experimental and bugs may exist.")

        if len(self.W) > 0:
            logger.warning(
                "Only Orthogonal Learners are currently supported with 'W', as Meta-Learners neccesitate 'W' in final CATE learner. "
                "If you don't care about 'W' features being used in final CATE model, add it to 'X' argument insead."
            )

    def auto_nuisance_functions(
        self,
        *,
        flaml_Y_kwargs: dict | None = None,
        flaml_T_kwargs: dict | None = None,
        use_ray: bool = False,
        use_spark: bool = False,
    ):
        """
        Leverages AutoML to find optimal nuisance functions/regression & propensity models for use in EconML CATE estimators.

        Sets the `model_Y_X_W`, `model_Y_X_W_T`, and `model_T_X_W` attributes to the fitted nuisance functions.

        Parameters
        ----------
        flaml_Y_kwargs
            The keyword arguments for the FLAML AutoML search for the outcome model. Default implies the base parameters in CamlBase.
        flaml_T_kwargs
            The keyword arguments for the FLAML AutoML search for the treatment model. Default implies the base parameters in CamlBase.
        use_ray
            A boolean indicating whether to use Ray for parallel processing.
        use_spark
            A boolean indicating whether to use Spark for parallel processing.

        Examples
        --------
        ```{python}
        flaml_Y_kwargs = {
            "n_jobs": -1,
            "time_budget": 10,
            "verbose": 0
        }

        flaml_T_kwargs = {
            "n_jobs": -1,
            "time_budget": 10,
            "verbose": 0
        }

        caml_obj.auto_nuisance_functions(
            flaml_Y_kwargs=flaml_Y_kwargs,
            flaml_T_kwargs=flaml_T_kwargs,
            use_ray=False,
            use_spark=False,
        )

        print(caml_obj.model_Y_X_W)
        print(caml_obj.model_Y_X_W_T)
        print(caml_obj.model_T_X_W)
        ```
        """
        if use_ray and not _HAS_RAY:
            raise ImportError(
                "Ray is not installed. Please install Ray to use it for parallel processing."
            )

        if use_spark and not _HAS_PYSPARK:
            raise ImportError(
                "PySpark is not installed. Please install PySpark optional dependencies via `pip install caml[pyspark]`."
            )

        self.model_Y_X_W = self._run_auto_nuisance_functions(
            outcome=self._Y,
            features=[self._X, self._W],
            discrete_outcome=self.discrete_outcome,
            flaml_kwargs=flaml_Y_kwargs,
            use_ray=use_ray,
            use_spark=use_spark,
        )
        self.model_Y_X_W_T = self._run_auto_nuisance_functions(
            outcome=self._Y,
            features=[self._X, self._W, self._T],
            discrete_outcome=self.discrete_outcome,
            flaml_kwargs=flaml_Y_kwargs,
            use_ray=use_ray,
            use_spark=use_spark,
        )
        self.model_T_X_W = self._run_auto_nuisance_functions(
            outcome=self._T,
            features=[self._X, self._W],
            discrete_outcome=self.discrete_treatment,
            flaml_kwargs=flaml_T_kwargs,
            use_ray=use_ray,
            use_spark=use_spark,
        )

        self._nuisances_fitted = True

    def fit_validator(
        self,
        *,
        cate_estimators: list[str] = [
            "LinearDML",
            "CausalForestDML",
            "NonParamDML",
            "SparseLinearDML-2D",
            "DRLearner",
            "ForestDRLearner",
            "LinearDRLearner",
            "DomainAdaptationLearner",
            "SLearner",
            "TLearner",
            "XLearner",
        ],
        additional_cate_estimators: list[tuple[str, BaseCateEstimator]] = [],
        ensemble: bool = False,
        rscorer_kwargs: dict = {},
        use_ray: bool = False,
        ray_remote_func_options_kwargs: dict = {},
        validation_size: float = 0.2,
        test_size: float = 0.2,
        sample_size: float = 1.0,
        n_jobs: int = -1,
    ):
        """
        Fits the CATE models on the training set and evaluates them & ensembles based on the validation set.

        Sets the `validation_estimator` attribute to the best fitted EconML estimator and `cate_estimators` attribute to all the fitted CATE models.

        Parameters
        ----------
        cate_estimators
            The list of CATE estimators to fit and ensemble. Default implies all available models as defined by class.
        additional_cate_estimators
            The list of additional CATE estimators to fit and ensemble
        ensemble
            The boolean indicating whether to ensemble the CATE models & score.
        rscorer_kwargs
            The keyword arguments for the econml.score.RScorer object.
        use_ray
            A boolean indicating whether to use Ray for parallel processing.
        ray_remote_func_options_kwargs
            The keyword arguments for the Ray remote function options.
        validation_size
            The fraction of the dataset to use for model scoring via RScorer.
        test_size
            The fraction of the dataset to hold out for final evaluation in the `validate()` method.
        sample_size
            The fraction of the datasets to use. Useful for quick testing when dataframe is large. Defaults implies full training data.
        n_jobs
            The number of parallel jobs to run.

        Examples
        --------
        ```{python}
        from econml.dr import LinearDRLearner

        rscorer_kwargs = {
            "cv": 3,
            "mc_iters": 3,
        }
        cate_estimators = ["LinearDML", "NonParamDML", "CausalForestDML"]
        additional_cate_estimators = [
            (
                "LinearDRLearner",
                LinearDRLearner(
                    model_propensity=caml_obj.model_T_X_W,
                    model_regression=caml_obj.model_Y_X_W_T,
                    discrete_outcome=caml_obj.discrete_outcome,
                    cv=3,
                    random_state=0,
                ),
            )
        ]

        caml_obj.fit_validator(
            cate_estimators=cate_estimators,
            additional_cate_estimators=additional_cate_estimators,
            rscorer_kwargs=rscorer_kwargs,
            validation_size=0.2,
            test_size=0.2
        )

        print(caml_obj.validation_estimator)
        print(caml_obj.cate_estimators)
        ```
        """
        if not self._nuisances_fitted:
            raise RuntimeError(
                "`find_nuissance_functions()` method must be called prior to estimating CATE models."
            )

        if use_ray and not _HAS_RAY:
            raise ImportError(
                "Ray is not installed. Please install Ray to use it for parallel processing."
            )

        self._split_data(
            validation_size=validation_size,
            test_size=test_size,
            sample_fraction=sample_size,
        )
        self.cate_estimators = self._get_cate_estimators(
            cate_estimators=cate_estimators,
            additional_cate_estimators=additional_cate_estimators,
        )
        (self._validation_estimator, self._rscorer) = (
            self._fit_and_ensemble_cate_estimators(
                rscorer_kwargs=rscorer_kwargs,
                use_ray=use_ray,
                ray_remote_func_options_kwargs=ray_remote_func_options_kwargs,
                n_jobs=n_jobs,
                ensemble=ensemble,
            )
        )

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
        n_groups
            The number of quantile based groups used to calculate calibration scores.
        n_bootstrap
            The number of boostrap samples to run when calculating confidence bands.
        estimator
            The estimator to validate. Default implies the best estimator from the validation set.
        print_full_report
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
            logger.error(
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
            logger.warning(
                "Some of the validation results suggest that the model may not have found statistically significant heterogeneity. Please closely look at the validation results and consider retraining with new configurations."
            )
        else:
            logger.info(
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

    def predict(
        self,
        *,
        X: pandas.DataFrame | np.ndarray | None = None,
        T0: int = 0,
        T1: int = 1,
        T: pandas.DataFrame | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predicts the CATE based on the fitted final estimator for either the internal dataset or provided Data.

        For binary treatments, the CATE is the estimated effect of the treatment and for a continuous treatment, the CATE is the estimated effect of a one-unit increase in the treatment.
        This can be modified by setting the T0 and T1 parameters to the desired treatment levels.

        Parameters
        ----------
        X
            The DataFrame containing the features (X) for which CATE needs to be predicted.
            If not provided, defaults to the internal dataset.
        T0
            Base treatment for each sample.
        T1
            Target treatment for each sample.
        T
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
        cate_predictions
            The CATE predictions for which summary statistics will be generated.
            If not provided, defaults to internal CATE predictions generated by `predict()` method with X=None.

        Returns
        -------
        pandas.DataFrame | pandas.Series
            The summary statistics for the CATE predictions.

        Examples
        --------
        ```{python}
        caml_obj.summarize()
        ```
        """
        if cate_predictions is None:
            _cate_predictions = self._cate_predictions
            cate_predictions_df = pandas.DataFrame.from_dict(_cate_predictions)
        else:
            _cate_predictions = cate_predictions
            cate_predictions_df = pandas.DataFrame(
                cate_predictions, columns=["cate_predictions"]
            )

        return cate_predictions_df.describe()

    def _get_cate_estimators(
        self,
        *,
        cate_estimators: list[str],
        additional_cate_estimators: list[tuple[str, BaseCateEstimator]],
    ):
        """
        Create model grid for CATE models to be fitted and ensembled.

        Sets the `_cate_models` internal attribute to the list of CATE models to fit and ensemble.

        Parameters
        ----------
        cate_estimators
            The list of CATE models to fit and ensemble.
        additional_cate_estimators
            The list of additional CATE models to fit and ensemble.
        """
        _cate_estimators = []
        for est in cate_estimators:
            model_tuple = model_bank.get_cate_model(
                est,
                self.model_Y_X_W,
                self.model_T_X_W,
                self.model_Y_X_W_T,
                self.discrete_treatment,
                self.discrete_outcome,
                self.seed,
            )
            if model_tuple is None:
                pass
            else:
                _cate_estimators.append(model_tuple)

        return _cate_estimators + additional_cate_estimators

    def _fit_and_ensemble_cate_estimators(
        self,
        *,
        rscorer_kwargs: dict,
        use_ray: bool,
        ray_remote_func_options_kwargs: dict,
        n_jobs: int = -1,
        ensemble: bool = False,
    ):
        """
        Fits the CATE estimators and, optionally, ensembles them.

        Parameters
        ----------
        rscorer_kwargs
            The keyword arguments for the econml.score.RScorer object.
        use_ray
            A boolean indicating whether to use Ray for parallel processing.
        ray_remote_func_options_kwargs
            The keyword arguments for the Ray remote function options.
        n_jobs
            The number of parallel jobs to run. Default implies -1 (all CPUs).
        ensemble
            Whether to ensemble the fitted CATE models.

        Returns
        -------
        econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
            The best fitted EconML estimator.
        econml.score.RScorer
            The fitted RScorer object.
        """
        Y_train, T_train, X_train, W_train = (
            self._data_splits["Y_train"],
            self._data_splits["T_train"],
            self._data_splits["X_train"],
            self._data_splits["W_train"],
        )

        Y_val, T_val, X_val, W_val = (
            self._data_splits["Y_val"],
            self._data_splits["T_val"],
            self._data_splits["X_val"],
            self._data_splits["W_val"],
        )

        def fit_model(name, model, use_ray=False, ray_remote_func_options_kwargs={}):
            if isinstance(model, _OrthoLearner):
                model.use_ray = use_ray
                model.ray_remote_func_options_kwargs = ray_remote_func_options_kwargs
                use_W = True
            else:
                use_W = False
            try:
                if use_W:
                    fitted_model = model.fit(
                        Y=Y_train,
                        T=T_train,
                        X=X_train,
                        W=W_train if W_train.shape[1] > 0 else None,
                    )
                else:
                    if len(self.W) > 0:
                        logger.warning(
                            f"Non-Orthogonal Learners ({name}) are not supported with 'W'. Skipping model."
                        )
                        fitted_model = None
                    else:
                        fitted_model = model.fit(Y=Y_train, T=T_train, X=X_train)
            except AttributeError as e:
                if (
                    str(e)
                    == "This method can only be used with single-dimensional continuous treatment or binary categorical treatment."
                ):
                    logger.warning(
                        f"Multi-dimensional discrete treatment is not supported for {name}. Skipping model."
                    )
                    fitted_model = None
                else:
                    raise e
            return name, fitted_model

        if use_ray:
            ray.init(ignore_reinit_error=True)

            models = [
                fit_model(
                    name,
                    model,
                    use_ray=True,
                    ray_remote_func_options_kwargs=ray_remote_func_options_kwargs,
                )
                for name, model in self.cate_estimators
            ]
        elif n_jobs == 1:
            models = [fit_model(name, model) for name, model in self.cate_estimators]
        else:
            models = Parallel(n_jobs=n_jobs)(
                delayed(fit_model)(name, model) for name, model in self.cate_estimators
            )

        models = [m for m in models if m[1] is not None]

        base_rscorer_settings = {
            "cv": 3,
            "mc_iters": 3,
            "mc_agg": "median",
            "random_state": self.seed,
        }

        if rscorer_kwargs is not None:
            base_rscorer_settings.update(rscorer_kwargs)

        rscorer = RScorer(  # BUG: RScorer does not work with discrete outcomes. See monkey patch below.
            model_y=self.model_Y_X_W,
            model_t=self.model_T_X_W,
            discrete_treatment=self.discrete_treatment,
            **base_rscorer_settings,
        )

        rscorer.fit(
            y=Y_val,
            T=T_val,
            X=X_val,
            W=W_val if W_val.shape[1] > 0 else None,
            discrete_outcome=self.discrete_outcome,
        )
        if ensemble:
            ensemble_estimator, ensemble_score, estimator_scores = rscorer.ensemble(
                [mdl for _, mdl in models], return_scores=True
            )
            estimator_scores = list(estimator_scores)
            estimator_scores.append(ensemble_score)
            models.append(("ensemble", ensemble_estimator))
            self.cate_estimators.append(("ensemble", ensemble_estimator))
        else:
            _, _, estimator_scores = rscorer.best_model(
                [mdl for _, mdl in models], return_scores=True
            )

        estimator_score_dict = dict(
            zip([n[0] for n in models], estimator_scores, strict=False)
        )
        best_estimator = models[np.nanargmax(estimator_scores)][0]

        logger.info(f"Best Estimator: {best_estimator}")
        logger.info(f"Estimator RScores: {estimator_score_dict}")

        return models[np.nanargmax(estimator_scores)][1], rscorer

    def __str__(self):
        """
        Returns a string representation of the CamlCATE object.

        Returns
        -------
        str
            A string containing information about the CamlCATE object, including data backend, number of observations, UUID, outcome variable, discrete outcome, treatment variable, discrete treatment, features/confounders, random seed, nuissance models (if fitted), and final estimator (if available).
        """
        summary = (
            "================== CamlCATE Object ==================\n"
            + f"Data Backend: {self._data_backend}\n"
            + f"No. of Observations: {self._Y.shape[0]:,}\n"
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
                f"Nuissance Model Y_X: {self.model_Y_X_W}\n"
                + f"Propensity/Nuissance Model T_X: {self.model_T_X_W}\n"
                + f"Regression Model Y_X_T: {self.model_Y_X_W_T}\n"
            )

        if self._final_estimator is not None:
            summary += f"Final Estimator: {self._final_estimator}\n"

        return summary


# Monkey patching Rscorer (Fixed in EconML PR - https://github.com/py-why/EconML/pull/927)
def patched_fit(
    self, y, T, X=None, W=None, sample_weight=None, groups=None, discrete_outcome=False
):
    if X is None:
        raise ValueError("X cannot be None for the RScorer!")

    self.lineardml_ = LinearDML(
        model_y=self.model_y,
        model_t=self.model_t,
        cv=self.cv,
        discrete_treatment=self.discrete_treatment,
        discrete_outcome=discrete_outcome,
        categories=self.categories,
        random_state=self.random_state,
        mc_iters=self.mc_iters,
        mc_agg=self.mc_agg,
    )
    self.lineardml_.fit(
        y,
        T,
        X=None,
        W=np.hstack([v for v in [X, W] if v is not None]),
        sample_weight=sample_weight,
        groups=groups,
        cache_values=True,
    )
    self.base_score_ = self.lineardml_.score_
    self.dx_ = X.shape[1]
    return self


RScorer.fit = patched_fit
