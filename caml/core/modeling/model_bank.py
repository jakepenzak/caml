import logging

from econml._cate_estimator import BaseCateEstimator
from econml.dml import CausalForestDML, LinearDML, NonParamDML, SparseLinearDML
from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner
from econml.metalearners import DomainAdaptationLearner, SLearner, TLearner, XLearner
from flaml import AutoML
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from typeguard import typechecked

logger = logging.getLogger(__name__)


@typechecked
def get_cate_model(
    model: str,
    mod_Y_X: BaseEstimator,
    mod_T_X: BaseEstimator,
    mod_Y_X_T: BaseEstimator,
    discrete_treatment: bool,
    discrete_outcome: bool,
    random_state: int | None = None,
    flaml_kwargs: dict = {},
) -> tuple[str, BaseCateEstimator] | None:
    """
    Returns the CATE model object given the model name and the models for Y|X, T|X, and Y|X,T.

    Parameters
    ----------
    model : str
        The name of the CATE model to use. Options are: LinearDML, CausalForestDML, NonParamDML, AutoNonParamDML, SparseLinearDML-2D, DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner-2D, DomainAdaptationLearner, SLearner, TLearner, XLearner
    mod_Y_X : BaseEstimator
        The model for Y|X.
    mod_T_X : BaseEstimator
        The model for T|X.
    mod_Y_X_T : BaseEstimator
        The model for Y|X,T.
    discrete_treatment : bool
        Whether the treatment is discrete.
    discrete_outcome : bool
        Whether the outcome is discrete.
    random_state : int
        The random state to use.
    flaml_kwargs : dict, optional
        The kwargs to pass to the Automl model if using AutoNonParamDML. Defaults to base settings.

    Returns
    -------
    tuple[str, BaseCateEstimator] | None
        The name of the model and the model object.
    """

    valid_models = [
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
    ]

    if model == "LinearDML":
        return model, LinearDML(
            model_y=mod_Y_X,
            model_t=mod_T_X,
            discrete_treatment=discrete_treatment,
            discrete_outcome=discrete_outcome,
            cv=3,
            random_state=random_state,
        )

    elif model == "CausalForestDML":
        return model, CausalForestDML(
            model_y=mod_Y_X,
            model_t=mod_T_X,
            discrete_treatment=discrete_treatment,
            discrete_outcome=discrete_outcome,
            cv=3,
            random_state=random_state,
        )

    elif model == "NonParamDML":
        if discrete_outcome:
            logger.warning("Discrete outcomes not yet supported for NonParamDML")
            return None
        else:
            return model, NonParamDML(
                model_y=mod_Y_X,
                model_t=mod_T_X,
                model_final=mod_Y_X_T,
                discrete_treatment=discrete_treatment,
                discrete_outcome=discrete_outcome,
                cv=3,
                random_state=random_state,
            )

    elif model == "AutoNonParamDML":
        base_settings = {
            "n_jobs": -1,
            "log_file_name": "",
            "seed": random_state,
            "time_budget": 120,
            "early_stop": "True",
            "eval_method": "cv",
            "n_splits": 3,
            "starting_points": "static",
            "estimator_list": [
                "lgbm",
                "rf",
                "xgboost",
                "extra_tree",
                "xgb_limitdepth",
            ],
            "task": "regression",
            "metric": "mse",
        }

        base_settings.update(flaml_kwargs)

        automl = AutoML(**base_settings)
        return model, NonParamDML(
            model_y=mod_Y_X,
            model_t=mod_T_X,
            model_final=automl,
            discrete_treatment=discrete_treatment,
            discrete_outcome=discrete_outcome,
            cv=3,
            random_state=random_state,
        )

    elif model == "SparseLinearDML-2D":
        return model, SparseLinearDML(
            model_y=mod_Y_X,
            model_t=mod_T_X,
            discrete_treatment=discrete_treatment,
            discrete_outcome=discrete_outcome,
            featurizer=PolynomialFeatures(degree=2),
            cv=3,
            random_state=random_state,
        )

    elif model in (
        "DRLearner",
        "ForestDRLearner",
        "LinearDRLearner",
        "SparseLinearDRLearner-2D",
        "DomainAdaptationLearner",
        "SLearner",
        "TLearner",
        "XLearner",
    ):
        if not discrete_treatment:
            logger.warning(f"Continuous treatments not supported for {model}!")
            return None

        elif model == "ForestDRLearner":
            return model, ForestDRLearner(
                model_propensity=mod_T_X,
                model_regression=mod_Y_X_T,
                discrete_outcome=discrete_outcome,
                cv=3,
                random_state=random_state,
            )

        elif model == "LinearDRLearner":
            return model, LinearDRLearner(
                model_propensity=mod_T_X,
                model_regression=mod_Y_X_T,
                discrete_outcome=discrete_outcome,
                cv=3,
                random_state=random_state,
            )

        elif model == "SparseLinearDRLearner-2D":
            return model, SparseLinearDRLearner(
                model_regression=mod_Y_X_T,
                model_propensity=mod_T_X,
                discrete_outcome=discrete_outcome,
                featurizer=PolynomialFeatures(degree=2),
                cv=3,
                random_state=random_state,
            )

        elif model in (
            "DomainAdaptationLearner",
            "SLearner",
            "TLearner",
            "XLearner",
            "DRLearner",
        ):
            if discrete_outcome:
                logger.warning(f"Discrete outcomes not yet supported for {model}!")

            elif model == "DomainAdaptationLearner":
                return model, DomainAdaptationLearner(
                    models=mod_Y_X_T,
                    final_models=mod_Y_X_T,
                    propensity_model=mod_T_X,
                )

            elif model == "SLearner":
                return model, SLearner(overall_model=mod_Y_X_T)

            elif model == "TLearner":
                return model, TLearner(models=mod_Y_X_T)

            elif model == "XLearner":
                return model, XLearner(
                    models=mod_Y_X_T, propensity_model=mod_T_X, cate_models=mod_Y_X_T
                )
            elif model == "DRLearner":
                return model, DRLearner(
                    model_propensity=mod_T_X,
                    model_regression=mod_Y_X_T,
                    model_final=mod_Y_X_T,
                    discrete_outcome=discrete_outcome,
                    cv=3,
                    random_state=random_state,
                )
    else:
        logger.warning(
            f"Model {model} not recognized. Please choose from: {valid_models}"
        )
        return None
