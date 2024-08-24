# from ..utils import descriptors
from ._base import CamlBase


class CamlDML(CamlBase):
    pass


# # TODO: Add validator decorators to methods.
# class CamlDML(CamlBase):
#     """
#     The CamlDML class represents a Double Machine Learning (DML) implementation for estimating...
#     average treatment effects (ATE), conditional average treatment effects (CATE), group average treatment effects (GATE),
#     etc.

#     This class... TODO

#     Parameters
#     ----------
#     df:
#         The input DataFrame representing the data for the EchoCATE instance.
#     uuid:
#         The str representing the column name for the universal identifier code (eg, ehhn)
#     y:
#         The str representing the column name for the outcome variable.
#     t:
#         The str representing the column name(s) for the treatment variable(s).
#     X:
#         The str (if unity) or list of feature names representing the custom feature set. Defaults to None.
#     model_y:
#         The nuissance model to be used for predicting the outcome. Defaults to HistGradientBoostingRegressor.
#     model_t:
#         The nuissance model to be used for predicting the treatment. Defaults to HistGradientBoostingClassifier.
#     discrete_treatment:
#         A boolean indicating whether the treatment is discrete or continuous. Defaults to True.
#     spark:
#         The SparkSession object used for connecting to Ibis when `df` is a pyspark.sql.DataFrame.
#         Defaults to None.

#     Attributes
#     ----------
#     df : pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | Table
#         The input DataFrame representing the data for the EchoCATE instance.
#     uuid: str
#         The str representing the column name for the universal identifier code (eg, ehhn)
#     y: str
#         The str representing the column name for the outcome variable.
#     t: str
#         The str representing the column name(s) for the treatment variable(s).
#     X: List[str] | str | None
#         The str (if unity) or list/tuple of feature names representing the custom feature set.
#     model_y: RegressorMixin | ClassifierMixin
#         The nuissance model to be used for predicting the outcome.
#     model_t: RegressorMixin | ClassifierMixin
#         The nuissance model to be used for predicting the treatment.
#     discrete_treatment: bool
#         A boolean indicating whether the treatment is discrete or continuous.
#     spark: SparkSession
#         The SparkSession object used for connecting to Ibis when `df` is a pyspark.sql.DataFrame.
#     _ibis_connection: ibis.client.Client
#         The Ibis client object representing the backend connection to Ibis.
#     _ibis_df: Table
#         The Ibis table expression representing the DataFrame connected to Ibis.
#     _table_name: str
#         The name of the temporary table/view created for the DataFrame in Ibis.
#     _Y: Table
#         The outcome variable data as ibis table.
#     _T: Table
#         The treatment variable data as ibis table.
#     _X: Table
#         The feature set data as ibis table.
#     _estimator: CausalForestDML
#         The fitted EconML estimator object.
#     """

#     # df = descriptors.ValidDataFrame(strict=True)
#     # Y = descriptors.ValidString(strict=True)
#     # T = descriptors.ValidString(strict=True)
#     # X = descriptors.ValidFeatureList(strict=False)
#     # uuid = descriptors.ValidString(strict=False)
#     # model_y = descriptors.ValidSklearnModel(strict=True)
#     # model_t = descriptors.ValidSklearnModel(strict=True)
#     # discrete_treatment = descriptors.ValidBoolean(strict=True)
#     # discrete_target = descriptors.ValidBoolean(strict=True)
#     # spark = descriptors.ValidSparkSession(strict=False)

#     # __slots__ = [
#     #     "_ibis_connection",
#     #     "_ibis_df",
#     #     "_table_name",
#     #     "_Y",
#     #     "_T",
#     #     "_X",
#     #     "_estimator",
#     # ]

#     def __init__(
#         self,
#         df: pandas.DataFrame | polars.DataFrame | pyspark.sql.DataFrame | Table,
#         Y: str,
#         T: str,
#         X: str | List[str] | None = None,
#         W: str | List[str] | None = None,
#         uuid: str | None = None,
#         model_y: RegressorMixin | ClassifierMixin = HistGradientBoostingRegressor(
#             max_depth=3,
#             max_iter=500,
#         ),
#         model_t: RegressorMixin | ClassifierMixin = HistGradientBoostingClassifier(
#             max_depth=3,
#             max_iter=500,
#         ),
#         discrete_treatment: bool = True,
#         discrete_outcome: bool = False,
#         spark: SparkSession | None = None,
#     ):
#         self.df = df
#         self.uuid = uuid
#         self.Y = Y
#         self.T = T
#         self.X = X
#         self.W = W
#         self.model_y = model_y
#         self.model_t = model_t
#         self.discrete_treatment = discrete_treatment
#         self.discrete_outcome = discrete_outcome
#         self.spark = spark

#         self._ibis_connector()

#     def fit(
#         self,
#         estimator: str = "CausalForestDML",
#         return_estimator: bool = False,
#         **kwargs,
#     ):
#         """
#         Fits the econometric model to learn the CATE function.

#         Sets the _Y, _T, and _X internal attributes to the data of the outcome, treatment, and feature set,
#         respectively. Additionally, sets the _estimator internal attribute to the fitted EconML estimator object.

#         Parameters
#         ----------
#         estimator:
#             The estimator to use for fitting the CATE function. Defaults to 'CausalForestDML'. Currently,
#             only this option is available.
#         return_estimator:
#             Set to True to recieve the estimator object back after fitting. Defaults to False.
#         **kwargs:
#             Additional keyword arguments to pass to the EconML estimator.

#         Returns
#         -------
#         econml.dml.causal_forest.CausalForestDML:
#             The fitted EconML CausalForestDML estimator object if `return_estimator` is True.
#         """

#         self._Y = self._ibis_df.select(self.Y)
#         self._T = self._ibis_df.select(self.T)
#         if self.X is None:
#             self._X = None
#         else:
#             self._X = self._ibis_df.select(self.X)

#         if estimator.lower() == "CausalForestDML".lower():
#             self._estimator = self._fit_causal_forest_dml(**kwargs)
#         else:
#             raise NotImplementedError(
#                 f"Estimator {estimator} not supported. Please use 'CausalForestDML'."
#             )

#         if return_estimator:
#             return self._estimator

#     def predict(
#         self,
#         out_of_sample_df: pandas.DataFrame
#         | polars.DataFrame
#         | pyspark.sql.DataFrame
#         | Table
#         | None = None,
#         ci: int = 90,
#         return_predictions: bool = False,
#         append_predictions: bool = False,
#     ):
#         """
#         Predicts the CATE given feature set.

#         Returns
#         -------
#         tuple:
#             A tuple containing the predicted CATE, standard errors, lower bound, and upper bound if `return_predictions` is True.
#         """

#         if out_of_sample_df is None:
#             X = self._X.execute().to_numpy()
#         else:
#             # oos = EchoCATE(oos_df)
#             # oos.get_features()
#             raise NotImplementedError("Out-of-sample prediction is not supported yet.")

#         effect_inference = self._estimator.effect_inference(X)

#         uuids = self._ibis_df[self.uuid].execute().to_numpy()
#         predictions = effect_inference.pred
#         standard_errors = effect_inference.pred_stderr
#         cv = stats.norm.ppf(1 - (1 - ci / 100) / 2)
#         lower_bound = predictions - cv * standard_errors
#         upper_bound = predictions + cv * standard_errors

#         data_dict = {
#             self.uuid: uuids,
#             "predictions": predictions,
#             "standard_errors": standard_errors,
#             "lower_bound": lower_bound,
#             "upper_bound": upper_bound,
#         }

#         if append_predictions:
#             results_df = self._create_internal_ibis_table(data_dict=data_dict)
#             self._ibis_df = self._ibis_df.join(
#                 results_df, predicates=self.uuid, how="inner"
#             )

#         if return_predictions:
#             return data_dict

#     def optimize(self):
#         """
#         Optimizes a households treatment based on CATE predictions. Only applicable when
#         vector of treatments includes more than 1 mutually exlusive treatment.

#         Returns
#         -------
#             None
#         """
#         return

#     def rank(self):
#         """
#         Ranks households based on the those with the highest estimated CATE.

#         Returns
#         -------
#             None
#         """
#         return

#     # TODO: Add in pretty dataframe of summary results in place of standard EconML output
#     def summarize(self):
#         """
#         Provides population summary of treatment effects, including Average Treatment Effects (ATEs)
#         and Conditional Average Treatement Effects (CATEs).

#         Returns
#         -------
#         econml.utilities.Summary:
#             Population summary of the results.
#         """

#         return self._estimator.summary()

#     def _fit_causal_forest_dml(self, **kwargs):
#         """
#         Fits the CausalForestDML model to learn the CATE function.

#         Parameters
#         ----------
#         **kwargs:
#             Additional keyword arguments to pass to the EconML CausalForestDML estimator.

#         Returns
#         -------
#         econml.dml.causal_forest.CausalForestDML:
#             The fitted EconML CausalForestDML estimator object.
#         """

#         base_kwargs = {
#             "discrete_treatment": self.discrete_treatment,
#             "discrete_outcome": self.discrete_outcome,
#             "model_y": self.model_y,
#             "model_t": self.model_t,
#             "criterion": "mse",
#             "cv": 5,
#             "mc_iters": 3,
#             "drate": True,
#             "min_samples_leaf": 150,
#             "n_estimators": 400,
#             "random_state": 8451,
#             "max_depth": None,
#         }

#         for key in kwargs:
#             base_kwargs[key] = kwargs[key]

#         estimator = CausalForestDML(**base_kwargs)

#         Y = self._Y.execute().to_numpy().ravel()
#         T = self._T.execute().to_numpy().ravel()
#         if self._X is None:
#             X = None
#         else:
#             X = self._X.execute().to_numpy()

#         estimator.fit(Y=Y, T=T, X=X, cache_values=True)

#         return estimator
