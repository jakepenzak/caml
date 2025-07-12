# pyright: reportArgumentType=false
import logging
from dataclasses import dataclass

from econml._cate_estimator import BaseCateEstimator
from econml.dml import CausalForestDML, LinearDML, NonParamDML, SparseLinearDML
from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner
from econml.metalearners import DomainAdaptationLearner, SLearner, TLearner, XLearner
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


available_estimators = [
    "LinearDML",
    "CausalForestDML",
    "NonParamDML",
    "SparseLinearDML-2D",
    "DRLearner",
    "ForestDRLearner",
    "LinearDRLearner",
    # "DomainAdaptationLearner",
    "SLearner",
    "TLearner",
    "XLearner",
]


@dataclass
class AutoCateEstimator:
    name: str
    estimator: BaseCateEstimator


def get_cate_estimator(
    model: str,
    model_Y: BaseEstimator,
    model_T: BaseEstimator,
    model_regression: BaseEstimator,
    discrete_treatment: bool,
    discrete_outcome: bool,
    random_state: int | None = None,
) -> AutoCateEstimator | None:
    """
    Returns the CATE model object given the model name and the models for Y|X, T|X, and Y|X,T.

    Parameters
    ----------
    model : str
        The name of the CATE model to use. Options are: LinearDML, CausalForestDML, NonParamDML, AutoNonParamDML, SparseLinearDML-2D, DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner-2D, DomainAdaptationLearner, SLearner, TLearner, XLearner
    model_Y : BaseEstimator
        The model for Y|X,W.
    model_T : BaseEstimator
        The model for T|X,W.
    model_regression : BaseEstimator
        The model for Y|X,T,W.
    discrete_treatment : bool
        Whether the treatment is discrete.
    discrete_outcome : bool
        Whether the outcome is discrete.
    random_state : int | None
        The random state to use.

    Returns
    -------
    AutoCateEstimator | None
        The name of the model and the model object.
    """
    if model == "LinearDML":
        return AutoCateEstimator(
            name=model,
            estimator=LinearDML(
                model_y=model_Y,
                model_t=model_T,
                discrete_treatment=discrete_treatment,
                discrete_outcome=discrete_outcome,
                cv=3,
                random_state=random_state,
            ),
        )
    elif model == "CausalForestDML":
        return AutoCateEstimator(
            name=model,
            estimator=CausalForestDML(
                model_y=model_Y,
                model_t=model_T,
                discrete_treatment=discrete_treatment,
                discrete_outcome=discrete_outcome,
                cv=3,
                random_state=random_state,
            ),
        )
    elif model == "NonParamDML":
        if discrete_outcome:
            logger.warning("Discrete outcomes not yet supported for NonParamDML")
            return None
        else:
            return AutoCateEstimator(
                name=model,
                estimator=NonParamDML(
                    model_y=model_Y,
                    model_t=model_T,
                    model_final=model_regression,
                    discrete_treatment=discrete_treatment,
                    discrete_outcome=discrete_outcome,
                    cv=3,
                    random_state=random_state,
                ),
            )
    elif model == "SparseLinearDML-2D":
        return AutoCateEstimator(
            name=model,
            estimator=SparseLinearDML(
                model_y=model_Y,
                model_t=model_T,
                discrete_treatment=discrete_treatment,
                discrete_outcome=discrete_outcome,
                featurizer=PolynomialFeatures(degree=2),
                cv=3,
                random_state=random_state,
            ),
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
            return AutoCateEstimator(
                name=model,
                estimator=ForestDRLearner(
                    model_propensity=model_T,
                    model_regression=model_regression,
                    discrete_outcome=discrete_outcome,
                    cv=3,
                    random_state=random_state,
                ),
            )
        elif model == "LinearDRLearner":
            return AutoCateEstimator(
                name=model,
                estimator=LinearDRLearner(
                    model_propensity=model_T,
                    model_regression=model_Y,
                    discrete_outcome=discrete_outcome,
                    cv=3,
                    random_state=random_state,
                ),
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
                return None

            elif model == "DomainAdaptationLearner":
                return AutoCateEstimator(
                    name=model,
                    estimator=DomainAdaptationLearner(
                        models=model_regression,
                        final_models=model_regression,
                        propensity_model=model_T,
                    ),
                )
            elif model == "SLearner":
                return AutoCateEstimator(
                    name=model,
                    estimator=SLearner(overall_model=model_regression),
                )
            elif model == "TLearner":
                return AutoCateEstimator(
                    name=model,
                    estimator=TLearner(models=model_regression),
                )
            elif model == "XLearner":
                return AutoCateEstimator(
                    name=model,
                    estimator=XLearner(
                        models=model_regression,
                        propensity_model=model_T,
                        cate_models=model_regression,
                    ),
                )
            elif model == "DRLearner":
                return AutoCateEstimator(
                    name=model,
                    estimator=DRLearner(
                        model_propensity=model_T,
                        model_regression=model_regression,
                        model_final=model_regression,
                        discrete_outcome=discrete_outcome,
                        cv=3,
                        random_state=random_state,
                    ),
                )
    else:
        logger.warning(
            f"Model {model} not recognized. Please choose from: {available_estimators}"
        )
        raise ValueError(
            f"Model {model} not recognized. Please choose from: {available_estimators}"
        )
