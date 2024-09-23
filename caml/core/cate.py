from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas
from econml._cate_estimator import BaseCateEstimator
from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML
from econml.score import EnsembleCateEstimator, RScorer
from econml.validate.drtester import DRTester
from ibis.common.exceptions import IbisTypeError
from joblib import Parallel, delayed

from ..utils import cls_typechecked
from . import model_bank
from ._base import CamlBase

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
    flexibility for advanced users. The class is designed to be used with `pandas`, `polars`, `pyspark`, or `ibis` backends to
    provide a level of extensibility & interoperability across different data processing frameworks.

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
    df : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.expr.types.Table
        The input DataFrame representing the data for the CamlCATE instance.
    Y : str
        The str representing the column name for the outcome variable.
    T : str
        The str representing the column name(s) for the treatment variable(s).
    X : list[str] | str | None
        The str (if unity) or list of feature names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATE.
    uuid : str | None
        The str representing the column name for the universal identifier code (eg, ehhn). Default implies index for joins.
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
    df : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.expr.types.Table
        The input DataFrame representing the data for the CamlCATE instance.
    Y : str
        The str representing the column name for the outcome variable.
    T : str
        The str representing the column name(s) for the treatment variable(s).
    X : list[str] | str
        The str (if unity) or list/tuple of feature names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATE.
    uuid : str
        The str representing the column name for the universal identifier code (eg, ehhn)
    discrete_treatment : bool
        A boolean indicating whether the treatment is discrete/categorical or continuous.
    discrete_outcome : bool
        A boolean indicating whether the outcome is binary or continuous.
    validation_estimator : econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object for validation.
    final_estimator : econml._cate_estimator.BaseCateEstimator | econml.score.EnsembleCateEstimator
        The fitted EconML estimator object on the entire dataset after validation.
    dataframe : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.expr.types.Table
        The input DataFrame with any modifications (e.g., predictions or rank orderings) made by the class returned to the original backend.
    model_Y_X: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable.
    model_Y_X_T: sklearn.base.BaseEstimator
        The fitted nuisance function for the outcome variable with treatment variable.
    model_T_X: sklearn.base.BaseEstimator
        The fitted nuisance function for the treatment variable.
    _ibis_connection: ibis.client.Client
        The Ibis client object representing the backend connection to Ibis.
    _ibis_df: ibis.expr.types.Table
        The Ibis table expression representing the DataFrame connected to Ibis.
    _table_name: str
        The name of the temporary table/view created for the DataFrame in the backend.
    _spark: pyspark.sql.SparkSession
        The Spark session object if the DataFrame is a Spark DataFrame.
    _Y: ibis.expr.types.Table
        The outcome variable data as ibis table.
    _T: ibis.expr.types.Table
        The treatment variable data as ibis table.
    _X: ibis.expr.types.Table
        The feature/confounder set data as ibis table.
    _X_T: ibis.expr.types.Table
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
        df: pandas.DataFrame
        | polars.DataFrame
        | pyspark.sql.DataFrame
        | ibis.expr.types.Table,
        Y: str,
        T: str,
        X: str | list[str],
        *,
        uuid: str | None = None,
        discrete_treatment: bool = True,
        discrete_outcome: bool = False,
        seed: int | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)

        self.df = df
        self.uuid = uuid
        self.Y = Y
        self.T = T
        self.X = X
        self.discrete_treatment = discrete_treatment
        self.discrete_outcome = discrete_outcome
        self.seed = seed
        self._spark = None

        self._ibis_connector()

        if self.uuid is None:
            self._ibis_df = self._ibis_df.mutate(
                uuid=ibis.row_number().over(ibis.window())
            )
            self.uuid = "uuid"

        self._Y = self._ibis_df.select(self.Y)
        self._T = self._ibis_df.select(self.T)
        self._X = self._ibis_df.select(self.X)
        self._X_T = self._ibis_df.select(self.X + [self.T])

        self._nuisances_fitted = False
        self._validation_estimator = None
        self._final_estimator = None

        if not self.discrete_treatment:
            logger.warning("Validation for continuous treatments is not supported yet.")

        if self.discrete_outcome:
            logger.error("Binary outcomes are not supported yet.")
            raise ValueError("Binary outcomes are not supported yet.")

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

        Sets the `model_Y_X`, `model_Y_X_T`, and `model_T_X` internal attributes to the fitted nuisance functions.

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

        self.model_Y_X = self._run_auto_nuisance_functions(
            outcome=self._Y,
            features=self._X,
            discrete_outcome=self.discrete_outcome,
            flaml_kwargs=flaml_Y_kwargs,
            use_ray=use_ray,
            use_spark=use_spark,
        )
        self.model_Y_X_T = self._run_auto_nuisance_functions(
            outcome=self._Y,
            features=self._X_T,
            discrete_outcome=self.discrete_outcome,
            flaml_kwargs=flaml_Y_kwargs,
            use_ray=use_ray,
            use_spark=use_spark,
        )
        self.model_T_X = self._run_auto_nuisance_functions(
            outcome=self._T,
            features=self._X,
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
        self._get_cate_models(
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
            model_regression=self.model_Y_X_T,
            model_propensity=self.model_T_X,
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

        if Y_test.shape[1] == 1:
            Y_test = Y_test.to_numpy().ravel()
            Y_train = Y_train.to_numpy().ravel()
        else:
            Y_test = Y_test.to_numpy()
            Y_train = Y_train.to_numpy()

        if T_test.shape[1] == 1:
            T_test = T_test.to_numpy().ravel()
            T_train = T_train.to_numpy().ravel()
        else:
            T_test = T_test.to_numpy()
            T_train = T_train.to_numpy()

        X_test = X_test.to_numpy()
        X_train = X_train.to_numpy()

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

        Y, T, X = self._Y.execute(), self._T.execute(), self._X.execute()

        if Y.shape[1] == 1:
            Y = Y.to_numpy().ravel()

        if T.shape[1] == 1:
            T = T.to_numpy().ravel()

        if isinstance(self._final_estimator, EnsembleCateEstimator):
            for estimator in self._final_estimator._cate_models:
                estimator.fit(
                    Y=Y,
                    T=T,
                    X=X.to_numpy(),
                )
        else:
            self._final_estimator.fit(
                Y=Y,
                T=T,
                X=X.to_numpy(),
            )

    def predict(
        self,
        *,
        out_of_sample_df: pandas.DataFrame
        | polars.DataFrame
        | pyspark.sql.DataFrame
        | ibis.expr.types.Table
        | None = None,
        out_of_sample_uuid: str | None = None,
        return_predictions: bool = False,
        join_predictions: bool = True,
        T0: int = 0,
        T1: int = 1,
    ):
        """
        Predicts the CATE based on the fitted final estimator for either the internal dataframe or an out-of-sample dataframe.

        For binary treatments, the CATE is the estimated effect of the treatment and for a continuous treatment, the CATE is the estimated effect of a one-unit increase in the treatment.
        This can be modified by setting the T0 and T1 parameters to the desired treatment levels.

        Parameters
        ----------
        out_of_sample_df: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.expr.types.Table | None
            The out-of-sample DataFrame to make predictions on.
        out_of_sample_uuid: str | None
            The column name for the universal identifier code (eg, ehhn) in the out-of-sample DataFrame.
        return_predictions: bool
            A boolean indicating whether to return the predicted CATE.
        join_predictions: bool
            A boolean indicating whether to join the predicted CATE to the original DataFrame within the class.
        T0: int
            Base treatment for each sample.
        T1: int
            Target treatment for each sample.

        Returns
        -------
        np.ndarray | DataFrame
            The predicted CATE values if return_predictions is set to True.

        Examples
        --------
        >>> caml.predict(join_predictions=True) # Joins the predicted CATE values to the original DataFrame.
        >>> caml.dataframe # Returns the DataFrame to original backend with the predicted CATE values joined.
        """

        assert (
            return_predictions or join_predictions
        ), "Either return_predictions or join_predictions must be set to True."

        assert self._final_estimator, "The final estimator must be fitted first before making predictions. Please run the fit() method with final_estimator=True."

        if out_of_sample_df is None:
            X = self._X.execute()
            uuids = self._ibis_df[self.uuid].execute()
            uuid_col = self.uuid
        else:
            input_df = self._create_internal_ibis_table(df=out_of_sample_df)
            if join_predictions:
                if out_of_sample_uuid is None:
                    try:
                        uuids = input_df[self.uuid].execute()
                        uuid_col = self.uuid
                    except IbisTypeError:
                        raise ValueError(
                            "The `uuid` column must be provided in the out-of-sample DataFrame to join predictions and the `out_of_sample_uuid` argument must be set to the string name of the column."
                        )
                else:
                    uuids = input_df[out_of_sample_uuid].execute()
                    uuid_col = out_of_sample_uuid
            X = input_df.select(self.X).execute()

        if self.discrete_treatment:
            num_categories = self._T.distinct().count().execute()
            data_dict = {}
            for c in range(1, num_categories):
                data_dict[f"cate_predictions_{c}"] = self._final_estimator.effect(
                    X, T0=0, T1=c
                )

            if join_predictions:
                data_dict[uuid_col] = uuids
                results_df = self._create_internal_ibis_table(data_dict=data_dict)
                if out_of_sample_df is None:
                    self._ibis_df = self._ibis_df.join(
                        results_df, predicates=uuid_col, how="inner"
                    )
                else:
                    final_df = input_df.join(
                        results_df, predicates=uuid_col, how="inner"
                    )
                    return self._return_ibis_dataframe_to_original_backend(
                        ibis_df=final_df, backend=input_df._find_backend().name
                    )

            if return_predictions:
                return data_dict
        else:
            cate_predictions = self._final_estimator.effect(X, T0=T0, T1=T1)

            data_dict = {"cate_predictions_1": cate_predictions}

            if join_predictions:
                data_dict[uuid_col] = uuids
                results_df = self._create_internal_ibis_table(data_dict=data_dict)
                if out_of_sample_df is None:
                    self._ibis_df = self._ibis_df.join(
                        results_df, predicates=uuid_col, how="inner"
                    )
                else:
                    final_df = input_df.join(
                        results_df, predicates=uuid_col, how="inner"
                    )
                    return self._return_ibis_dataframe_to_original_backend(
                        ibis_df=final_df, backend=input_df._find_backend().name
                    )

            if return_predictions:
                return data_dict

    def rank_order(
        self,
        *,
        out_of_sample_df: pandas.DataFrame
        | polars.DataFrame
        | pyspark.sql.DataFrame
        | ibis.expr.types.Table
        | None = None,
        return_rank_order: bool = False,
        join_rank_order: bool = True,
        treatment_category: int = 1,
    ):
        """
        Ranks orders households based on the predicted CATE values for either the internal dataframe or an out-of-sample dataframe.

        Parameters
        ----------
        out_of_sample_df: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.expr.types.Table | None
            The out-of-sample DataFrame to rank order.
        return_rank_order: bool
            A boolean indicating whether to return the rank ordering.
        join_rank_order: bool
            A boolean indicating whether to join the rank ordering to the original DataFrame within the class.
        treatment_category: int
            The treatment category, in the case of categorical treatments, to rank order the households based on. Default implies the first category.

        Returns
        -------
        np.ndarray | DataFrame
            The rank ordering values if return_rank_order is set to True.

        Examples
        --------
        >>> caml.rank_order(join_rank_order=True) # Joins the rank ordering to the original DataFrame.
        >>> caml.dataframe # Returns the DataFrame to original backend with the rank ordering values joined.
        """

        assert (
            return_rank_order or join_rank_order
        ), "Either return_rank_order or join_rank_order must be set to True."
        assert (
            self._ibis_connection.name != "polars"
        ), "Rank ordering is not supported for polars DataFrames."

        if out_of_sample_df is None:
            df = self._ibis_df
        else:
            df = self._create_internal_ibis_table(df=out_of_sample_df)

        assert (
            "cate_predictions" in c for c in df.columns
        ), "CATE predictions must be present in the DataFrame to rank order. Please call the predict() method first with join_predictions=True."

        window = ibis.window(
            order_by=ibis.desc(df[f"cate_predictions_{treatment_category}"])
        )
        df = df.mutate(cate_ranking=ibis.row_number().over(window))

        if return_rank_order:
            return df.select("cate_ranking").execute().to_numpy()

        elif join_rank_order:
            if out_of_sample_df is None:
                self._ibis_df = df.order_by("cate_ranking")
            else:
                final_df = self._return_ibis_dataframe_to_original_backend(
                    ibis_df=df.order_by("cate_ranking"),
                    backend=df._find_backend().name,
                )
                return final_df

    def summarize(
        self,
        *,
        out_of_sample_df: pandas.DataFrame
        | polars.DataFrame
        | pyspark.sql.DataFrame
        | ibis.expr.types.Table
        | None = None,
        treatment_category: int = 1,
    ):
        """
        Provides population summary statistics for the CATE predictions for either the internal dataframe or an out-of-sample dataframe.

        Parameters
        ----------
        out_of_sample_df: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.expr.types.Table | None
            The out-of-sample DataFrame to summarize.
        treatment_category: int
            The treatment level, in the case of categorical treatments, to summarize the CATE predictions for. Default implies the first category.

        Returns
        -------
        pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | ibis.expr.types.Table
            The summary statistics for the CATE predictions.

        Examples
        --------
        >>> caml.summarize() # Summarizes the CATE predictions for the internal DataFrame.
        """

        if out_of_sample_df is None:
            df = self._ibis_df
        else:
            df = self._create_internal_ibis_table(df=out_of_sample_df)

        assert (
            "cate_predictions" in c for c in df.columns
        ), "CATE predictions must be present in the DataFrame to summarize. Please call the predict() method first with join_predictions=True."

        column = df[f"cate_predictions_{treatment_category}"]

        cate_summary_statistics = df.aggregate(
            [
                column.mean().name("cate_mean"),
                column.sum().name("cate_sum"),
                column.std().name("cate_std"),
                column.min().name("cate_min"),
                column.max().name("cate_max"),
                column.count().name("count"),
            ]
        )

        return self._return_ibis_dataframe_to_original_backend(
            ibis_df=cate_summary_statistics
        )

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

        mod_Y_X = self.model_Y_X
        mod_T_X = self.model_T_X
        mod_Y_X_T = self.model_Y_X_T
        discrete_treatment = self.discrete_treatment
        discrete_outcome = self.discrete_outcome
        random_state = self.seed

        self._cate_models = []
        for cate_model in subset_cate_models:
            self._cate_models.append(
                model_bank.get_cate_model(
                    cate_model,
                    mod_Y_X,
                    mod_T_X,
                    mod_Y_X_T,
                    discrete_treatment,
                    discrete_outcome,
                    random_state,
                )
            )

        self._cate_models = self._cate_models + additional_cate_models

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

        Y_train, T_train, X_train = (
            self._data_splits["Y_train"],
            self._data_splits["T_train"],
            self._data_splits["X_train"],
        )

        Y_val, T_val, X_val = (
            self._data_splits["Y_val"],
            self._data_splits["T_val"],
            self._data_splits["X_val"],
        )

        if Y_train.shape[1] == 1:
            Y_train = Y_train.to_numpy().ravel()
            Y_val = Y_val.to_numpy().ravel()

        if T_train.shape[1] == 1:
            T_train = T_train.to_numpy().ravel()
            T_val = T_val.to_numpy().ravel()

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
            model_y=self.model_Y_X,
            model_t=self.model_T_X,
            discrete_treatment=self.discrete_treatment,
            **base_rscorer_settings,
        )

        rscorer.fit(Y=Y_val, T=T_val, X=X_val, discrete_outcome=self.discrete_outcome)

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
            + f"Data Backend: {self._ibis_connection.name}\n"
            + f"No. of Observations: {self._Y.count().execute()}\n"
            + f"UUID: {self.uuid}\n"
            + f"Outcome Variable: {self.Y}\n"
            + f"Discrete Outcome: {self.discrete_outcome}\n"
            + f"Treatment Variable: {self.T}\n"
            + f"Discrete Treatment: {self.discrete_treatment}\n"
            + f"Features/Confounders: {self.X}\n"
            + f"Random Seed: {self.seed}\n"
        )

        if self._nuisances_fitted:
            summary += (
                f"Nuissance Model Y_X: {self.model_Y_X}\n"
                + f"Propensity/Nuissance Model T_X: {self.model_T_X}\n"
                + f"Regression Model Y_X_T: {self.model_Y_X_T}\n"
            )

        if self._final_estimator is not None:
            summary += f"Final Estimator: {self._final_estimator}\n"

        return summary


# Monkey patching Rscorer
def patched_fit(
    self, Y, T, X=None, W=None, sample_weight=None, groups=None, discrete_outcome=False
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
        Y,
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
