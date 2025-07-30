# pyright: reportArgumentType=false
import logging
from dataclasses import dataclass

from econml._cate_estimator import BaseCateEstimator
from econml.dml import CausalForestDML, LinearDML, NonParamDML, SparseLinearDML
from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


@dataclass
class AutoCateEstimator:
    name: str
    estimator: BaseCateEstimator


AutoLinearDML = AutoCateEstimator(
    name="LinearDML", estimator=LinearDML(model_y="auto", model_t="auto", cv=3)
)

AutoCausalForestDML = AutoCateEstimator(
    name="CausalForestDML",
    estimator=CausalForestDML(model_y="auto", model_t="auto", cv=3),
)

AutoNonParamDML = AutoCateEstimator(
    name="NonParamDML",
    estimator=NonParamDML(model_y="auto", model_t="auto", model_final="auto", cv=3),
)

AutoSparseLinearDML = AutoCateEstimator(
    name="SparseLinearDML-2D",
    estimator=SparseLinearDML(
        model_y="auto", model_t="auto", featurizer=PolynomialFeatures(degree=2), cv=3
    ),
)

AutoDRLearner = AutoCateEstimator(
    name="DRLearner",
    estimator=DRLearner(
        model_propensity="auto", model_regression="auto", model_final="auto", cv=3
    ),
)

AutoForestDRLearner = AutoCateEstimator(
    name="ForestDRLearner",
    estimator=ForestDRLearner(model_propensity="auto", model_regression="auto", cv=3),
)

AutoLinearDRLearner = AutoCateEstimator(
    name="LinearDRLearner",
    estimator=LinearDRLearner(model_propensity="auto", model_regression="auto", cv=3),
)

AutoSLearner = AutoCateEstimator(
    name="SLearner", estimator=SLearner(overall_model="auto")
)

AutoTLearner = AutoCateEstimator(name="TLearner", estimator=TLearner(models="auto"))

AutoXLearner = AutoCateEstimator(
    name="XLearner",
    estimator=XLearner(models="auto", propensity_model="auto", cate_models="auto"),
)

available_estimators = {
    AutoLinearDML.name: AutoLinearDML,
    AutoCausalForestDML.name: AutoCausalForestDML,
    AutoNonParamDML.name: AutoNonParamDML,
    AutoSparseLinearDML.name: AutoSparseLinearDML,
    AutoDRLearner.name: AutoDRLearner,
    AutoForestDRLearner.name: AutoForestDRLearner,
    AutoLinearDRLearner.name: AutoLinearDRLearner,
    AutoSLearner.name: AutoSLearner,
    AutoTLearner.name: AutoTLearner,
    AutoXLearner.name: AutoXLearner,
}
