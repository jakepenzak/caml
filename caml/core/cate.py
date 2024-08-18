from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from econml._cate_estimator import BaseCateEstimator

import numpy as np
import pandas
import polars

try:
    import pyspark
except ImportError:
    pass
import logging

import ibis
import ray
from econml._ortho_learner import _OrthoLearner
from econml.dml import DML, CausalForestDML, LinearDML, NonParamDML
from econml.dr import DRLearner
from econml.metalearners import DomainAdaptationLearner, SLearner, TLearner, XLearner
from econml.score import EnsembleCateEstimator, RScorer
from econml.validate.drtester import DRTester
from flaml import AutoML
from ibis.common.exceptions import IbisTypeError
from ibis.expr.types.relations import Table
from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from ..utils import descriptors
from ._base import CamlBase

logger = logging.getLogger(__name__)


class CamlCATE(CamlBase):
    """
    The CamlCATE class represents an optimized implementation of Causal Machine Learning techniques for estimating
    highly accurate conditional average treatment effects (CATEs) and constucting CATE ensemble models.

    This class... TODO

    Parameters
    ----------
    df:
        The input DataFrame representing the data for the EchoCATE instance.
    Y:
        The str representing the column name for the outcome variable.
    T:
        The str representing the column name(s) for the treatment variable(s).
    X:
        The str (if unity) or list of feature names representing the heterogeneity feature set. Defaults to None.
    W:
        The str (if unity) or list of feature names representing the confounder feature set. Defaults to None.
    uuid:
        The str representing the column name for the universal identifier code (eg, ehhn). Defaults to None, which implies index for joins.
    discrete_treatment:
        A boolean indicating whether the treatment is discrete or continuous. Defaults to True.

    Attributes
    ----------
    df : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | Table
        The input DataFrame representing the data for the EchoCATE instance.=
    Y: str
        The str representing the column name for the outcome variable.
    T: str
        The str representing the column name(s) for the treatment variable(s).
    X: List[str] | str | None
        The str (if unity) or list/tuple of feature names representing the heterogeneity feature set.
    W: List[str] | str | None
        The str (if unity) or list/tuple of feature names representing the confounder feature set.
    uuid: str
        The str representing the column name for the universal identifier code (eg, ehhn)
    discrete_treatment: bool
        A boolean indicating whether the treatment is discrete or continuous.
    _ibis_connection: ibis.client.Client
        The Ibis client object representing the backend connection to Ibis.
    _ibis_df: Table
        The Ibis table expression representing the DataFrame connected to Ibis.
    _table_name: str
        The name of the temporary table/view created for the DataFrame in Ibis.
    _Y: Table
        The outcome variable data as ibis table.
    _T: Table
        The treatment variable data as ibis table.
    _X: Table
        The feature set data as ibis table.
    _estimator: CausalForestDML
        The fitted EconML estimator object.
    """

    df = descriptors.ValidDataFrame(strict=True)
    Y = descriptors.ValidString(strict=True)
    T = descriptors.ValidString(strict=True)
    X = descriptors.ValidFeatureList(strict=False)
    W = descriptors.ValidFeatureList(strict=False)
    uuid = descriptors.ValidString(strict=False)
    discrete_treatment = descriptors.ValidBoolean(strict=True)
    discrete_target = descriptors.ValidBoolean(strict=True)

    __slots__ = [
        "_spark",
        "_ibis_connection",
        "_ibis_df",
        "_table_name",
        "_Y",
        "_T",
        "_X",
        "_W",
        "_estimator",
        "_model_Y_X",
        "_model_T_X",
        "_model_Y_X_T",
        "_cate_models",
        "_best_estimator",
        "_rscorer",
        "_data_partitions",
        "_nuisances_fitted",
        "_validator_results",
        "_final_estimator",
    ]

    def __init__(
        self,
        df: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | Table,
        Y: str,
        T: str,
        X: str | List[str] | None = None,
        W: str | List[str] | None = None,
        uuid: str | None = None,
        discrete_treatment: bool = True,
        discrete_outcome: bool = False,
    ):
        self.df = df
        self.uuid = uuid
        self.Y = Y
        self.T = T
        self.X = X
        self.W = W
        self.discrete_treatment = discrete_treatment
        self.discrete_outcome = discrete_outcome
        self._spark = None

        self._ibis_connector()

        self._nuisances_fitted = False
        self._best_estimator = None
        self._final_estimator = None

    def find_nuisance_functions(
        self,
        *,
        automl_Y_kwargs: dict | None = None,
        automl_T_kwargs: dict | None = None,
        use_ray: bool = False,
        use_spark: bool = False,
    ):
        """
        TODO
        """
        self._Y = self._ibis_df.select(self.Y)
        self._T = self._ibis_df.select(self.T)
        self._X = self._ibis_df.select(self.X) if self.X is not None else None
        self._W = self._ibis_df.select(self.W) if self.W is not None else None

        self._model_Y_X, self._model_Y_X_T, self._model_T_X = (
            self._automl_nuisance_functions(
                automl_Y_kwargs=automl_Y_kwargs,
                automl_T_kwargs=automl_T_kwargs,
                use_ray=use_ray,
                use_spark=use_spark,
            )
        )

        self._nuisances_fitted = True

    def fit(
        self,
        *,
        subset_cate_models: List[str] = [
            "LinearDML",
            "NonParamDML",
            "DML-Lasso3d",
            "CausalForestDML",
            "XLearner",
            "DomainAdaptationLearner",
            "SLearner",
            "TLearner",
            "DRLearner",
        ],
        custom_cate_models: dict | None = None,
        rscorer_kwargs: dict | None = None,
        use_ray: bool = True,
        ray_remote_func_options_kwargs: dict = {},
        final_estimator: bool = False,
    ):
        """
        Fits the econometric model to learn the CATE function.

        Sets the _Y, _T, and _X internal attributes to the data of the outcome, treatment, and feature set,
        respectively. Additionally, sets the _estimator internal attribute to the fitted EconML estimator object.

        Parameters
        ----------
        estimator:
            The estimator to use for fitting the CATE function. Defaults to 'CausalForestDML'. Currently,
            only this option is available.
        automl_Y_kwargs:
            The settings to use for the AutoML model for the outcome. Defaults to None.
        automl_T_kwargs:
            The settings to use for the AutoML model for the treatment. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the EconML estimator.

        Returns
        -------
        econml.dml.causal_forest.CausalForestDML:
            The fitted EconML CausalForestDML estimator object if `return_estimator` is True.
        """

        assert self._nuisances_fitted, "find_nuissance_functions() method must be called first to find optimal nussiance functions for estimating CATE models."

        if final_estimator:
            assert (
                self._best_estimator is not None
            ), "The best estimator must be fitted first before fitting the final estimator."

            if isinstance(self._best_estimator, EnsembleCateEstimator):
                for estimator in self._best_estimator._cate_models:
                    estimator.fit(
                        Y=self._Y.execute().to_numpy().ravel(),
                        T=self._T.execute().to_numpy().ravel(),
                        X=self._X.execute().to_numpy(),
                    )
                self._final_estimator = self._best_estimator
            else:
                self._best_estimator.fit(
                    Y=self._Y.execute().to_numpy().ravel(),
                    T=self._T.execute().to_numpy().ravel(),
                    X=self._X.execute().to_numpy(),
                )
                self._final_estimator = self._best_estimator
        else:
            if custom_cate_models is None:
                self._get_cate_models(subset_cate_models=subset_cate_models)
            else:
                self._cate_models = custom_cate_models
            (
                self._best_estimator,
                self._rscorer,
                self._data_partitions,
            ) = self._fit_and_ensemble_cate_models(
                rscorer_settings=rscorer_kwargs,
                use_ray=use_ray,
                ray_remote_func_options_kwargs=ray_remote_func_options_kwargs,
            )

    def validate(
        self,
        *,
        estimator: BaseCateEstimator | None = None,
    ):
        """
        Validates the CATE model.

        Returns
        -------
            None
        """

        if estimator is None:
            estimator = self._best_estimator

        validator = DRTester(
            model_regression=self._model_Y_X_T,
            model_propensity=self._model_T_X,
            cate=estimator,
        )

        X_train, _, X_test, T_train, _, T_test, Y_train, _, Y_test = (
            self._data_partitions.values()
        )

        validator.fit_nuisance(
            X_test, T_test.astype(int), Y_test, X_train, T_train.astype(int), Y_train
        )

        res = validator.evaluate_all(X_test, X_train)

        # Check for insignificant results & warn user
        summary = res.summary()
        if np.array(summary[[c for c in summary.columns if "pval" in c]] > 0.1).any():
            logger.warn(
                "Some of the validation results suggest that the model has not found statistically significant heterogeneity. Please closely look at the validation results and consider retraining with new configurations."
            )

        self._validator_results = res

        return res

    def predict(
        self,
        *,
        out_of_sample_df: pandas.DataFrame
        | polars.DataFrame
        | pyspark.sql.DataFrame
        | Table
        | None = None,
        out_of_sample_uuid: str | None = None,
        return_predictions: bool = False,
        append_predictions: bool = False,
    ):
        """
        Predicts the CATE given feature set.

        Returns
        -------
        tuple:
            A tuple containing the predicted CATE, standard errors, lower bound, and upper bound if `return_predictions` is True.
        """

        assert (
            return_predictions or append_predictions
        ), "Either return_predictions or append_predictions must be set to True."

        assert (
            self._final_estimator is not None
        ), "The final estimator must be fitted first before making predictions. Please run the fit() method with final_estimator=True."

        if out_of_sample_df is None:
            X = self._X.execute().to_numpy()
            uuids = self._ibis_df[self.uuid].execute().to_numpy()
            uuid_col = self.uuid
        else:
            input_df = self._create_internal_ibis_table(df=out_of_sample_df)
            if append_predictions is True:
                if out_of_sample_uuid is None:
                    try:
                        uuids = input_df[self.uuid].execute().to_numpy()
                        uuid_col = self.uuid
                    except IbisTypeError:
                        raise ValueError(
                            "The `uuid` column must be provided in the out-of-sample DataFrame and the `out_of_sample_uuid` argument must be set to the string name of the column."
                        )
                else:
                    uuids = input_df[out_of_sample_uuid].execute().to_numpy()
                    uuid_col = out_of_sample_uuid
            X = input_df.select(self.X).execute().to_numpy()

        cate_predictions = self._best_estimator.effect(X)

        if append_predictions:
            data_dict = {
                uuid_col: uuids,
                "cate_predictions": cate_predictions,
            }
            results_df = self._create_internal_ibis_table(data_dict=data_dict)
            if out_of_sample_df is None:
                self._ibis_df = self._ibis_df.join(
                    results_df, predicates=uuid_col, how="inner"
                )
                return
            else:
                final_df = input_df.join(results_df, predicates=uuid_col, how="inner")
                return self._return_ibis_dataframe_to_original_backend(
                    ibis_df=final_df, backend=input_df._find_backend().name
                )

        if return_predictions:
            return cate_predictions

    def rank_order(
        self,
        *,
        out_of_sample_df: pandas.DataFrame
        | polars.DataFrame
        | pyspark.sql.DataFrame
        | Table
        | None = None,
        return_rank_order: bool = False,
        append_rank_order: bool = False,
    ):
        """
        Ranks households based on the those with the highest estimated CATE.

        Returns
        -------
            None
        """

        assert (
            return_rank_order or append_rank_order
        ), "Either return_rank_order or append_rank_order must be set to True."
        assert (
            self._ibis_connection.name != "polars"
        ), "Rank ordering is not supported for polars DataFrames."

        if out_of_sample_df is None:
            assert (
                "cate_predictions" in self._ibis_df.columns
            ), "CATE predictions must be present in the DataFrame to rank order. Please call the predict() method first with append_predictions=True."

            window = ibis.window(order_by=ibis.desc(self._ibis_df["cate_predictions"]))
            self._ibis_df = self._ibis_df.mutate(
                cate_ranking=ibis.row_number().over(window)
            )

            if return_rank_order:
                return self._ibis_df.select("cate_ranking").execute().to_numpy()

            elif append_rank_order:
                self._ibis_df = self._ibis_df.order_by("cate_ranking")

        else:
            input_df = self._create_internal_ibis_table(df=out_of_sample_df)
            assert (
                "cate_predictions" in input_df.columns
            ), "CATE predictions must be present in the DataFrame to rank order. Please call the predict() method first with append_predictions=True, passing the out_of_sample_dataframe."

            window = ibis.window(order_by=ibis.desc(input_df["cate_predictions"]))
            final_df = input_df.mutate(cate_ranking=ibis.row_number().over(window))

            if return_rank_order:
                return final_df.select("cate_ranking").execute().to_numpy()
            elif append_rank_order:
                return self._return_ibis_dataframe_to_original_backend(
                    ibis_df=final_df.order_by("cate_ranking"),
                    backend=input_df._find_backend().name,
                )

    def summarize(self):
        """
        Provides population summary of treatment effects, including Average Treatment Effects (ATEs)
        and Conditional Average Treatement Effects (CATEs).

        Returns
        -------
        econml.utilities.Summary:
            Population summary of the results.
        """

        assert (
            "cate_predictions" in self._ibis_df.columns
        ), "CATE predictions must be present in the DataFrame to summarize. Please call the predict() method first with append_predictions=True."

        column = self._ibis_df["cate_predictions"]

        cate_summary_statistics = self._ibis_df.aggregate(
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

    def _automl_nuisance_functions(
        self,
        *,
        automl_Y_kwargs: dict | None,
        automl_T_kwargs: dict | None,
        use_ray: bool,
        use_spark: bool,
    ):
        """
        Automatically selects the best nuisance models for the outcome and treatment.

        """

        automl_Y_X = AutoML()
        automl_Y_X_T = AutoML()
        automl_T_X = AutoML()

        base_settings = {
            "n_jobs": -1,
            "log_file_name": "",
            "seed": 123,
            "time_budget": 300,
            "early_stop": "True",
            "starting_points": "data",
            "estimator_list": ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"],
        }

        # Set default settings for AutoML Outcome model
        _automl_Y_kwargs = base_settings.copy()
        if self.discrete_outcome:
            _automl_Y_kwargs["task"] = "classification"
            _automl_Y_kwargs["metric"] = "log_loss"
        else:
            _automl_Y_kwargs["task"] = "regression"
            _automl_Y_kwargs["metric"] = "mse"

        # Set default settings for AutoML Treatment model
        _automl_T_kwargs = base_settings.copy()
        if self.discrete_treatment:
            _automl_T_kwargs["task"] = "classification"
            _automl_T_kwargs["metric"] = "log_loss"
        else:
            _automl_T_kwargs["task"] = "regression"
            _automl_T_kwargs["metric"] = "mse"

        if self._spark is not None or use_spark:
            _automl_Y_kwargs["use_spark"], _automl_T_kwargs["use_spark"] = True, True
            (
                _automl_Y_kwargs["n_concurrent_trials"],
                _automl_T_kwargs["n_concurrent_trials"],
            ) = 4, 4

        elif use_ray:
            _automl_Y_kwargs["use_ray"], _automl_T_kwargs["use_ray"] = True, True
            (
                _automl_Y_kwargs["n_concurrent_trials"],
                _automl_T_kwargs["n_concurrent_trials"],
            ) = 4, 4

        if automl_T_kwargs is not None:
            _automl_T_kwargs.update(automl_T_kwargs)

        if automl_Y_kwargs is not None:
            _automl_Y_kwargs.update(automl_Y_kwargs)

        # Fit the AutoML models
        X = self._X.execute().to_numpy()
        Y = self._Y.execute().to_numpy().ravel()
        T = self._T.execute().to_numpy().ravel()
        XT = np.concatenate((X, T.reshape(-1, 1)), axis=1)

        automl_Y_X.fit(X, Y, **_automl_Y_kwargs)
        automl_Y_X_T.fit(XT, Y, **_automl_Y_kwargs)
        automl_T_X.fit(X, T, **_automl_T_kwargs)

        model_Y_X = automl_Y_X.model.estimator
        model_Y_X_T = automl_Y_X_T.model.estimator
        model_T_X = automl_T_X.model.estimator

        return model_Y_X, model_Y_X_T, model_T_X

    def _get_cate_models(self, *, subset_cate_models: List[str]):
        """
        Create model grid for CATE models to be fitted and ensembled.
        """

        mod_Y_X = self._model_Y_X
        mod_T_X = self._model_T_X
        mod_Y_X_T = self._model_Y_X_T

        self._cate_models = [
            (
                "LinearDML",
                LinearDML(
                    model_y=mod_Y_X,
                    model_t=mod_T_X,
                    discrete_treatment=self.discrete_treatment,
                    discrete_outcome=self.discrete_outcome,
                    cv=3,
                ),
            ),
            (
                "NonParamDML",
                NonParamDML(
                    model_y=mod_Y_X,
                    model_t=mod_T_X,
                    model_final=mod_Y_X_T,
                    discrete_treatment=self.discrete_treatment,
                    discrete_outcome=self.discrete_outcome,
                    cv=3,
                ),
            ),
            (
                "DML-Lasso3d",
                DML(
                    model_y=mod_Y_X,
                    model_t=mod_T_X,
                    model_final=LassoCV(),
                    discrete_treatment=self.discrete_treatment,
                    discrete_outcome=self.discrete_outcome,
                    featurizer=PolynomialFeatures(degree=3),
                    cv=3,
                ),
            ),
            (
                "CausalForestDML",
                CausalForestDML(
                    model_y=mod_Y_X,
                    model_t=mod_T_X,
                    discrete_treatment=self.discrete_treatment,
                    discrete_outcome=self.discrete_outcome,
                    cv=3,
                ),
            ),
        ]
        if self.discrete_treatment:
            self._cate_models.append(
                (
                    "XLearner",
                    XLearner(
                        models=mod_Y_X_T,
                        cate_models=mod_Y_X_T,
                        propensity_model=mod_T_X,
                    ),
                )
            )
            self._cate_models.append(
                (
                    "DomainAdaptationLearner",
                    DomainAdaptationLearner(
                        models=mod_Y_X, final_models=mod_Y_X_T, propensity_model=mod_T_X
                    ),
                )
            )
            self._cate_models.append(("SLearner", SLearner(overall_model=mod_Y_X_T)))
            self._cate_models.append(("TLearner", TLearner(models=mod_Y_X_T)))
            self._cate_models.append(
                (
                    "DRLearner",
                    DRLearner(
                        model_propensity=mod_T_X,
                        model_regression=mod_Y_X_T,
                        model_final=mod_Y_X_T,
                        cv=3,
                    ),
                )
            )

        self._cate_models = [m for m in self._cate_models if m[0] in subset_cate_models]

    def _fit_and_ensemble_cate_models(
        self,
        *,
        rscorer_settings: dict | None,
        use_ray: bool,
        ray_remote_func_options_kwargs: dict,
    ):
        """
        Fits the CATE models and ensembles them.

        Returns
        -------
        """

        X = self._X.execute().to_numpy()
        Y = self._Y.execute().to_numpy().ravel()
        T = self._T.execute().to_numpy().ravel()

        X_int, X_test, T_int, T_test, Y_int, Y_test = train_test_split(
            X, T, Y, test_size=0.2
        )

        X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(
            X_int, T_int, Y_int, test_size=0.2
        )

        def fit_model(name, model, use_ray=False, ray_remote_func_options_kwargs=None):
            if isinstance(model, _OrthoLearner):
                model.use_ray = use_ray
                model.ray_remote_func_options_kwargs = ray_remote_func_options_kwargs
            if name == "CausalForestDML":
                return name, model.tune(Y=Y_train, T=T_train, X=X_train).fit(
                    Y=Y_train, T=T_train, X=X_train
                )
            return name, model.fit(Y=Y_train, T=T_train, X=X_train)

        if use_ray:
            ray.init(ignore_reinit_error=True)

            fit_model = ray.remote(fit_model).options(**ray_remote_func_options_kwargs)
            futures = [
                fit_model.remote(
                    name,
                    model,
                    use_ray=True,
                    ray_remote_func_options_kwargs=ray_remote_func_options_kwargs,
                )
                for name, model in self._cate_models
            ]
            models = ray.get(futures)
        else:
            models = Parallel(n_jobs=-1)(
                delayed(fit_model)(name, mdl) for name, mdl in self._cate_models
            )

        base_rscorer_settings = {
            "cv": 3,
            "mc_iters": 3,
            "mc_agg": "median",
        }

        if rscorer_settings is not None:
            base_rscorer_settings.update(rscorer_settings)

        rscorer = RScorer(
            model_y=self._model_Y_X,
            model_t=self._model_T_X,
            discrete_treatment=self.discrete_treatment,
            **base_rscorer_settings,
        )

        rscorer.fit(Y_val, T_val, X_val)

        ensemble_estimator, ensemble_score, estimator_scores = rscorer.ensemble(
            [mdl for _, mdl in models], return_scores=True
        )

        logger.info(f"Ensemble Estimator RScore: {ensemble_score}")
        logger.info(
            f"Inidividual Estimator RScores: {dict(zip([n[0] for n in models],estimator_scores))}"
        )

        # Choose best estimator
        def get_best_estimator(ensemble_estimator, ensemble_score, estimator_scores):
            if np.max(estimator_scores) >= ensemble_score:
                logger.info(
                    "The best estimator is greater than the ensemble estimator. Returning that individual estimator."
                )
                best_estimator = ensemble_estimator._cate_models[
                    np.argmax(estimator_scores)
                ]
            else:
                logger.info(
                    "The ensemble estimator is the best estimator, filtering models with weights less than 0.01."
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
                best_estimator = ensemble_estimator

            return best_estimator

        best_estimator = get_best_estimator(
            ensemble_estimator, ensemble_score, estimator_scores
        )

        data_partitions = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "T_train": T_train,
            "T_val": T_val,
            "T_test": T_test,
            "Y_train": Y_train,
            "Y_val": Y_val,
            "Y_test": Y_test,
        }

        return best_estimator, rscorer, data_partitions
