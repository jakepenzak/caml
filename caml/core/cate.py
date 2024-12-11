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

from ..utils import cls_typechecked
from ._base import CamlBase
from .utils import model_bank

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import polars

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

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


# TODO: Update docstrings to include for mathematical details.
@cls_typechecked
class CamlCATE(CamlBase):
    """
    The CamlCATE class represents an opinionated framework of Causal Machine Learning techniques for estimating
    highly accurate conditional average treatment effects (CATEs).

    This class is built on top of the EconML library and provides a high-level API for fitting, validating, and making inference with CATE models,
    with best practices built directly into the API. The class is designed to be easy to use and understand, while still providing
    flexibility for advanced users. The class is designed to be used with `pandas`, `polars`, or `pyspark` backends, which ultimately get
    converted to NumPy Arrays under the hood to provide a level of extensibility & interoperability across different data processing frameworks.

    The primary workflow for the CamlCATE class is as follows:

    1. Initialize the class with the input DataFrame and the necessary columns.
    2. Utilize AutoML to find the optimal nuisance functions or propensity/regression models to be utilized in the EconML estimators.
    3. Fit the CATE models on the training set and evaluate based on the validation set, then select the top performer/ensemble.
    4. Validate the fitted CATE model on the test set to check for generalization performance.
    5. Fit the final estimator on the entire dataset, after validation and testing.
    6. Predict the CATE based on the fitted final estimator for either the internal dataframe or an out-of-sample dataframe.
    7. Rank orders households based on the predicted CATE values for either the internal dataframe or an out-of-sample dataframe.
    8. Summarize population summary statistics for the CATE predictions for either the internal dataframe or an out-of-sample dataframe.


    For technical details on conditional average treatment effects, see:

     - CaML Documentation
     - [EconML documentation](https://econml.azurewebsites.net/)

     **Note**: All the standard assumptions of Causal Inference apply to this class (e.g., exogeneity/unconfoundedness, overlap, positivity, etc.).
        The class does not check for these assumptions and assumes that the user has already thought through these assumptions before using the class.

    **Outcome & Treatment Data Type Support Matrix**

    | Outcome     | Treatment   | Support     | Missing    |
    | ----------- | ----------- | ----------- | ---------- |
    | Continuous  | Binary      | ✅Full      | None       |
    | Continuous  | Continuous  | 🟡Partial   | Validation |
    | Continuous  | Categorical | ✅Full      | None       |
    | Binary      | Binary      | ❌Not yet   |            |
    | Binary      | Continuous  | ❌Not yet   |            |
    | Binary      | Categorical | ❌Not yet   |            |
    | Categorical | Binary      | ❌Not yet   |            |
    | Categorical | Continuous  | ❌Not yet   |            |
    | Categorical | Categorical | ❌Not yet   |            |

    Multi-dimensional outcomes and treatments are not yet supported.

    Parameters
    ----------
    df : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame
        The input DataFrame representing the data for the CamlCATE instance.
    Y : str
        The str representing the column name for the outcome variable.
    T : str
        The str representing the column name(s) for the treatment variable(s).
    X : list[str] | str | None
        The str (if unity) or list of feature names representing the feature set to be utilized for estimating heterogeneity/CATE.
    W : list[str] | str | None
        The str (if unity) or list of feature names representing the confounder/control feature set to be utilized only for nuisance function estimation where applicable.
    discrete_treatment : bool
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome : bool
        A boolean indicating whether the outcome is binary or continuous.
    seed : int | None
        The seed to use for the random number generator.
    verbose : int
        The verbosity level for logging. Default implies 1 (INFO). Set to 0 for no logging. Set to 2 for DEBUG.

    Attributes
    ----------
    df : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.Table
        The input DataFrame representing the data for the CamlCATE instance.
    Y : str
        The str representing the column name for the outcome variable.
    T : str
        The str representing the column name(s) for the treatment variable(s).
    X : list[str] | str
        The str (if unity) or list/tuple of feature names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATE and nuisance function estimation where applicable.
    W : list[str] | str
        The str (if unity) or list/tuple of feature names representing the confounder/control feature set to be utilized only for nuisance function estimation, where applicable. These will be included by default in Meta-Learners.
    discrete_treatment : bool
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome : bool
        A boolean indicating whether the outcome is binary or continuous.
    validation_estimator : econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object for validation.
    final_estimator : econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object on the entire dataset after validation.
    dataframe : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.Table
        The input DataFrame with any modifications (e.g., predictions or rank orderings) made by the class returned to the original backend.
    model_Y_X_W: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable.
    model_Y_X_W_T: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable with treatment variable.
    model_T_X_W: sklearn.base.BaseEstimator
        The fitted nuisance function for the treatment variable.
    _Y: ibis.Table
        The outcome variable data as ibis table.
    _T: ibis.Table
        The treatment variable data as ibis table.
    _X: ibis.Table
        The feature/confounder set data as ibis table.
    _X_T: ibis.Table
        The feature/confounder feature set and treatment variable data as ibis table.
    _nuisances_fitted: bool
        A boolean indicating whether the nuisance functions have been fitted.
    _validation_estimator: econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object for validation.
    _final_estimator: econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object for final predictions.
    _validator_results: econml.validate.EvaluationResults
        The results of the validation tests from DRTester.
    _cate_models: list[tuple[str, econml._cate_estimator.BaseCateEstimator]]
        The list of CATE models to fit and ensemble.
    _data_splits: dict[str, np.ndarray]
        The dictionary containing the training, validation, and test data splits.
    _rscorer: econml.score.RScorer
        The RScorer object for the validation estimator.

    Examples
    --------
    >>> from caml.core.cate import CamlCATE
    >>> from caml.extensions.synthetic_data import make_fully_heterogeneous_dataset
    >>> df, true_cates, true_ate = make_fully_heterogeneous_dataset(n_obs=1000, n_confounders=10, theta=10, seed=1)
    >>> df['uuid'] = df.index
    >>>  caml_obj= CamlCATE(df=df, Y="y", T="d", X=[c for c in df.columns if "X" in c], uuid="uuid", discrete_treatment=True, discrete_outcome=False, seed=1)
    >>>
    >>> # Standard pipeline
    >>> caml_obj.auto_nuisance_functions()
    >>> caml_obj.fit_validator()
    >>> caml_obj.validate(print_full_report=True)
    >>> caml_obj.fit_final()
    >>> caml_obj.predict(join_predictions=True)
    >>> caml_obj.rank_order(join_rank_order=True)
    >>> caml_obj.summarize()
    >>>
    >>> end_of_pipeline_results = caml_obj.dataframe
    >>> final_estimator = caml_obj.final_estimator # Can be saved for future inference.
    """

    def __init__(
        self,
        df: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame,
        Y: str,
        T: str,
        X: str | list[str],
        W: str | list[str],
        *,
        discrete_treatment: bool = True,
        discrete_outcome: bool = False,
        seed: int | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)

        self.df = df
        self.Y = Y
        self.T = T
        self.X = X
        self.W = W
        self.discrete_treatment = discrete_treatment
        self.discrete_outcome = discrete_outcome
        self.seed = seed

        self._data_backend = (
            "pandas"
            if isinstance(self.df, pandas.DataFrame)
            else "polars"
            if _HAS_POLARS and isinstance(self.df, polars.DataFrame)
            else "pyspark"
            if _HAS_PYSPARK
            and isinstance(self.df, (pyspark.sql.DataFrame, pyspark.pandas.DataFrame))
            else "unknown"
        )
        self._Y, self._T, self._X, self._W = self._dataframe_to_numpy()

        self._nuisances_fitted = False
        self._validation_estimator = None
        self._final_estimator = None
        self._cate_predictions = {}

        if not self.discrete_treatment:
            logger.warning("Validation for continuous treatments is not supported yet.")

        if self.discrete_outcome:
            logger.warning("Binary outcomes are experimental and bugs may exist.")

    def auto_nuisance_functions(
        self,
        *,
        flaml_Y_kwargs: dict | None = None,
        flaml_T_kwargs: dict | None = None,
        use_ray: bool = False,
        use_spark: bool = False,
    ):
        """
        Automatically finds the optimal nuisance functions for estimating EconML estimators.

        Sets the `model_Y_X_W`, `model_Y_X_W_T`, and `model_T_X_W` internal attributes to the fitted nuisance functions.

        Parameters
        ----------
        flaml_Y_kwargs: dict | None
            The keyword arguments for the FLAML AutoML search for the outcome model. Default implies the base parameters in CamlBase.
        flaml_T_kwargs: dict | None
            The keyword arguments for the FLAML AutoML search for the treatment model. Default implies the base parameters in CamlBase.
        use_ray: bool
            A boolean indicating whether to use Ray for parallel processing.
        use_spark: bool
            A boolean indicating whether to use Spark for parallel processing.

        Examples
        --------
        >>> flaml_Y_kwargs = {
        ...     "n_jobs": -1,
        ...     "time_budget": 300, # in seconds
        ...     }
        >>> flaml_T_kwargs = {
        ...     "n_jobs": -1,
        ...     "time_budget": 300,
        ...     }
        >>> caml_obj.auto_nuisance_functions(flaml_Y_kwargs=flaml_Y_kwargs, flaml_T_kwargs=flaml_T_kwargs)
        """

        if use_ray:
            assert _HAS_RAY, "Ray is not installed. Please install Ray to use it for parallel processing."

        if use_spark:
            assert _HAS_PYSPARK, "PySpark is not installed. Please install PySpark optional dependencies via `pip install caml[pyspark]`."

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
        subset_cate_models: list[str] = [
            "LinearDML",
            "CausalForestDML",
            "NonParamDML",
            "AutoNonParamDML",
            "SparseLinearDML-2D",
            "DRLearner",
            "ForestDRLearner",
            "LinearDRLearner",
            "SparseLinearDRLearner-2D",
            "DomainAdaptationLearner",
            "SLearner",
            "TLearner",
            "XLearner",
        ],
        additional_cate_models: list[tuple[str, BaseCateEstimator]] = [],
        rscorer_kwargs: dict = {},
        use_ray: bool = False,
        ray_remote_func_options_kwargs: dict = {},
        sample_fraction: float = 1.0,
        n_jobs: int = 1,
    ):
        """
        Fits the CATE models on the training set and evaluates them & ensembles based on the validation set.

        Sets the `_validation_estimator` and `_rscorer` internal attributes to the fitted EconML estimator and RScorer object.

        Parameters
        ----------
        subset_cate_models: list[str]
            The list of CATE models to fit and ensemble. Default implies all available models as defined by class.
        additional_cate_models: list[tuple[str, econml._cate_estimator.BaseCateEstimator]]
            The list of additional CATE models to fit and ensemble
        rscorer_kwargs: dict
            The keyword arguments for the econml.score.RScorer object.
        use_ray: bool
            A boolean indicating whether to use Ray for parallel processing.
        ray_remote_func_options_kwargs: dict
            The keyword arguments for the Ray remote function options.
        sample_fraction: float
            The fraction of the training data to use for fitting the CATE models. Default implies 1.0 (full training data).
        n_jobs: int
            The number of parallel jobs to run. Default implies 1 (no parallel jobs).

        Examples
        --------
        >>> rscorer_kwargs = {
        ...     "cv": 3,
        ...     "mc_iters": 3,
        ...     }
        >>> subset_cate_models = ["LinearDML", "NonParamDML", "CausalForestDML"]
        >>> additional_cate_models = [("XLearner", XLearner(models=caml_obj._model_Y_X_T, cate_models=caml_obj._model_Y_X_T, propensity_model=caml._model_T_X))]
        >>> caml_obj.fit_validator(subset_cate_models=subset_cate_models, additional_cate_models=additional_cate_models, rscorer_kwargs=rscorer_kwargs)
        """

        assert self._nuisances_fitted, "find_nuissance_functions() method must be called first to find optimal nussiance functions for estimating CATE models."

        if use_ray:
            assert _HAS_RAY, "Ray is not installed. Please install Ray to use it for parallel processing."

        self._split_data(
            validation_size=0.2, test_size=0.2, sample_fraction=sample_fraction
        )
        self._cate_models = self._get_cate_models(
            subset_cate_models=subset_cate_models,
            additional_cate_models=additional_cate_models,
        )
        (self._validation_estimator, self._rscorer) = (
            self._fit_and_ensemble_cate_models(
                rscorer_kwargs=rscorer_kwargs,
                use_ray=use_ray,
                ray_remote_func_options_kwargs=ray_remote_func_options_kwargs,
                n_jobs=n_jobs,
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
        Validates the fitted CATE models on the test set to check for generalization performance. Uses the DRTester class from EconML to obtain the Best
        Linear Predictor (BLP), Calibration, AUTOC, and QINI. See [EconML documentation](https://econml.azurewebsites.net/_autosummary/econml.validate.DRTester.html) for more details.
        In short, we are checking for the ability of the model to find statistically significant heterogeneity in a "well-calibrated" fashion.

        Sets the `_validator_results` internal attribute to the results of the DRTester class.

        Parameters
        ----------
        n_groups: int
            The number of quantile based groups used to calculate calibration scores.
        n_bootstrap: int
            The number of boostrap samples to run when calculating confidence bands.
        estimator: econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
            The estimator to validate. Default implies the best estimator from the validation set.
        print_full_report: bool
            A boolean indicating whether to print the full validation report.

        Examples
        --------
        >>> caml_obj.validate(print_full_report=True) # Prints the full validation report.
        """
        plt.style.use("ggplot")

        if estimator is None:
            estimator = self._validation_estimator

        if not self.discrete_treatment:
            logger.error("Validation for continuous treatments is not supported yet.")
            raise ValueError(
                "Validation for continuous treatments is not supported yet."
            )

        validator = DRTester(
            model_regression=self.model_Y_X_W_T,
            model_propensity=self.model_T_X_W,
            cate=estimator,
            cv=3,
        )

        X_test, T_test, Y_test = (
            self._data_splits["X_test"],
            self._data_splits["T_test"],
            self._data_splits["Y_test"],
        )

        X_train, T_train, Y_train = (
            self._data_splits["X_train"],
            self._data_splits["T_train"],
            self._data_splits["Y_train"],
        )

        validator.fit_nuisance(
            X_test, T_test.astype(int), Y_test, X_train, T_train.astype(int), Y_train
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

        self._validator_results = res

    def fit_final(self):
        """
        Fits the final estimator on the entire dataset, after validation and testing.

        Sets the `_final_estimator` internal attribute to the fitted EconML estimator.

        Examples
        --------
        >>> caml_obj.fit_final() # Fits the final estimator on the entire dataset.
        """

        assert (
            self._validation_estimator
        ), "The best estimator must be fitted first before fitting the final estimator."

        self._final_estimator = copy.deepcopy(self._validation_estimator)

        Y, T, X = self._Y, self._T, self._X

        if isinstance(self._final_estimator, EnsembleCateEstimator):
            for estimator in self._final_estimator._cate_models:
                estimator.fit(
                    Y=Y,
                    T=T,
                    X=X,
                )
        else:
            self._final_estimator.fit(
                Y=Y,
                T=T,
                X=X,
            )

    def predict(
        self,
        *,
        X: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame = None,
        T0: int = 0,
        T1: int = 1,
    ):
        """
        Predicts the CATE based on the fitted final estimator for either the internal dataset or provided Data.

        For binary treatments, the CATE is the estimated effect of the treatment and for a continuous treatment, the CATE is the estimated effect of a one-unit increase in the treatment.
        This can be modified by setting the T0 and T1 parameters to the desired treatment levels.

        Parameters
        ----------
        X : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame
            The DataFrame containing the features (X) for which CATE needs to be predicted.
            If not provided, defaults to the internal dataset.
        T0 : int
            Base treatment for each sample.
        T1 : int
            Target treatment for each sample.

        Returns
        -------
        np.ndarray
            The predicted CATE values if return_predictions is set to True.

        Examples
        --------
        >>> caml.predict(return_as_dataframe=True)
        """

        assert self._final_estimator, "The final estimator must be fitted first before making predictions. Please run the fit() method with final_estimator=True."

        if X is None:
            _X = self._X
        else:
            _X = X

        cate_predictions = self._final_estimator.effect(_X, T0=T0, T1=T1)

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
            If not provided, defaults to internal CATE predictions genered by `predict()` method with X=None.

        Returns
        -------
        pandas.DataFrame | pandas.Series
            The summary statistics for the CATE predictions.

        Examples
        --------
        >>> caml.summarize() # Summarizes the CATE predictions for the internal DataFrame.
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

    def _get_cate_models(
        self,
        *,
        subset_cate_models: list[str],
        additional_cate_models: list[tuple[str, BaseCateEstimator]],
    ):
        """
        Create model grid for CATE models to be fitted and ensembled.

        Sets the `_cate_models` internal attribute to the list of CATE models to fit and ensemble.

        Parameters
        ----------
        subset_cate_models: list[str]
            The list of CATE models to fit and ensemble.
        additional_cate_models: list[tuple[str, econml._cate_estimator.BaseCateEstimator]]
            The list of additional CATE models to fit and ensemble.
        """

        _cate_models = []
        for cate_model in subset_cate_models:
            _cate_models.append(
                model_bank.get_cate_model(
                    cate_model,
                    self.model_Y_X_W,
                    self.model_T_X_W,
                    self.model_Y_X_W_T,
                    self.discrete_treatment,
                    self.discrete_outcome,
                    self.seed,
                )
            )

        return _cate_models + additional_cate_models

    def _fit_and_ensemble_cate_models(
        self,
        *,
        rscorer_kwargs: dict,
        use_ray: bool,
        ray_remote_func_options_kwargs: dict,
        n_jobs: int = -1,
    ):
        """
        Fits the CATE models and ensembles them.

        Parameters
        ----------
        rscorer_kwargs: dict
            The keyword arguments for the econml.score.RScorer object.
        use_ray: bool
            A boolean indicating whether to use Ray for parallel processing.
        ray_remote_func_options_kwargs: dict
            The keyword arguments for the Ray remote function options.
        n_jobs: int
            The number of parallel jobs to run. Default implies -1 (all CPUs).

        Returns
        -------
        econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
            The best fitted EconML estimator.
        econml.score.RScorer
            The fitted RScorer object.
        """

        Y_train, T_train, X_train, W_train = (  # noqa: F841
            self._data_splits["Y_train"],
            self._data_splits["T_train"],
            self._data_splits["X_train"],
            self._data_splits["W_train"],
        )

        Y_val, T_val, X_val, W_val = (  # noqa: F841
            self._data_splits["Y_val"],
            self._data_splits["T_val"],
            self._data_splits["X_val"],
            self._data_splits["W_val"],
        )

        def fit_model(name, model, use_ray=False, ray_remote_func_options_kwargs={}):
            if isinstance(model, _OrthoLearner):
                model.use_ray = use_ray
                model.ray_remote_func_options_kwargs = ray_remote_func_options_kwargs
            return name, model.fit(Y=Y_train, T=T_train, X=X_train)

        if use_ray:
            ray.init(ignore_reinit_error=True)

            models = [
                fit_model(
                    name,
                    model,
                    use_ray=True,
                    ray_remote_func_options_kwargs=ray_remote_func_options_kwargs,
                )
                for name, model in self._cate_models
            ]
        elif n_jobs == 1:
            models = [fit_model(name, model) for name, model in self._cate_models]
        else:
            models = Parallel(n_jobs=n_jobs)(
                delayed(fit_model)(name, model) for name, model in self._cate_models
            )

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

        rscorer.fit(y=Y_val, T=T_val, X=X_val, discrete_outcome=self.discrete_outcome)

        ensemble_estimator, ensemble_score, estimator_scores = rscorer.ensemble(
            [mdl for _, mdl in models], return_scores=True
        )

        logger.info(f"Ensemble Estimator RScore: {ensemble_score}")
        logger.info(
            f"Inidividual Estimator RScores: {dict(zip([n[0] for n in models],estimator_scores))}"
        )

        # Choose best estimator
        def get_validation_estimator(
            ensemble_estimator, ensemble_score, estimator_scores
        ):
            if np.max(estimator_scores) >= ensemble_score:
                best_estimator = ensemble_estimator._cate_models[
                    np.argmax(estimator_scores)
                ]
                logger.info(
                    f"The best estimator is greater than the ensemble estimator. Returning that individual estimator: {best_estimator}"
                )
            else:
                logger.info(
                    "The ensemble estimator is the best estimator, filtering out models with weights less than 0.01."
                )
                estimator_weight_map = dict(
                    zip(ensemble_estimator._cate_models, ensemble_estimator._weights)
                )
                ensemble_estimator._cate_models = [
                    k for k, v in estimator_weight_map.items() if v > 0.01
                ]
                ensemble_estimator._weights = np.array(
                    [v for _, v in estimator_weight_map.items() if v > 0.01]
                )
                ensemble_estimator._weights = ensemble_estimator._weights / np.sum(
                    ensemble_estimator._weights
                )
                best_estimator = ensemble_estimator

            return best_estimator

        best_estimator = get_validation_estimator(
            ensemble_estimator, ensemble_score, estimator_scores
        )

        return best_estimator, rscorer

    def __str__(self):
        """
        Returns a string representation of the CamlCATE object.

        Returns
        -------
        summary : str
            A string containing information about the CamlCATE object, including data backend, number of observations, UUID, outcome variable, discrete outcome, treatment variable, discrete treatment, features/confounders, random seed, nuissance models (if fitted), and final estimator (if available).
        """

        summary = (
            "================== CamlCATE Object ==================\n"
            + f"Data Backend: {self._data_backend}\n"
            + f"No. of Observations: {self._Y.shape[0]}\n"
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


# Monkey patching Rscorer
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
