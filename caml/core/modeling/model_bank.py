import logging

from econml._cate_estimator import BaseCateEstimator
from econml.dml import CausalForestDML, LinearDML, NonParamDML, SparseLinearDML
from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner
from econml.metalearners import DomainAdaptationLearner, SLearner, TLearner, XLearner
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from typeguard import typechecked

logger = logging.getLogger(__name__)


valid_models = [
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
]


@typechecked
def get_cate_model(
    model: str,
    mod_Y_X: BaseEstimator,
    mod_T_X: BaseEstimator,
    mod_Y_X_T: BaseEstimator,
    discrete_treatment: bool,
    discrete_outcome: bool,
    random_state: int | None = None,
) -> tuple[str, BaseCateEstimator] | None:
    """
    Returns the CATE model object given the model name and the models for Y|X, T|X, and Y|X,T.

    Parameters
    ----------
    model
        The name of the CATE model to use. Options are: LinearDML, CausalForestDML, NonParamDML, AutoNonParamDML, SparseLinearDML-2D, DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner-2D, DomainAdaptationLearner, SLearner, TLearner, XLearner
    mod_Y_X
        The model for Y|X.
    mod_T_X
        The model for T|X.
    mod_Y_X_T
        The model for Y|X,T.
    discrete_treatment
        Whether the treatment is discrete.
    discrete_outcome
        Whether the outcome is discrete.
    random_state
        The random state to use.

    Returns
    -------
    tuple[str, BaseCateEstimator] | None
        The name of the model and the model object.
    """
    assert model in valid_models, "Model not in list of valid models"

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
