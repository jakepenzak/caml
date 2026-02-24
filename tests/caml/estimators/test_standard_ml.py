"""Tests for caml.estimators.standard_ml module."""

import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier, XGBRegressor

from caml.automl.search_space import SearchSpaceSpec
from caml.estimators.standard_ml import (
    AVAILABLE_STANDARD_ML_ESTIMATORS,
    BaseStandardMLEstimator,
    ElasticNetModel,
    ExtraTreesModel,
    LGBMModel,
    # LogisticRegressionModel, # Removing temporarily since we only use regression classes currently
    RandomForestModel,
    XGBoostLimitDepthModel,
    XGBoostModel,
)

pytestmark = pytest.mark.estimators


# ==============================================================================
# REGISTRY TESTS
# ==============================================================================


class TestAvailableStandardMLEstimators:
    """Test AVAILABLE_STANDARD_ML_ESTIMATORS registry."""

    def test_registry_exists(self):
        """Test that registry is populated."""
        assert AVAILABLE_STANDARD_ML_ESTIMATORS is not None
        assert isinstance(AVAILABLE_STANDARD_ML_ESTIMATORS, dict)

    def test_expected_models_registered(self):
        """Test that expected model keys are registered."""
        expected_keys = {
            "lightgbm",
            "xgboost",
            "xgboost_limitdepth",
            "random_forest",
            "extra_trees",
            # "logistic",
            "elastic_net",
        }
        assert set(AVAILABLE_STANDARD_ML_ESTIMATORS.keys()) == expected_keys

    def test_all_values_are_classes(self):
        """Test that all registry values are classes."""
        for model_class in AVAILABLE_STANDARD_ML_ESTIMATORS.values():
            assert isinstance(model_class, type)

    def test_all_inherit_from_base(self):
        """Test that all registered models inherit from BaseStandardMLEstimator."""
        for model_class in AVAILABLE_STANDARD_ML_ESTIMATORS.values():
            assert issubclass(model_class, BaseStandardMLEstimator)


# ==============================================================================
# BASE ESTIMATOR TESTS
# ==============================================================================


class TestBaseStandardMLEstimator:
    """Test BaseStandardMLEstimator ABC."""

    def test_cannot_instantiate_directly(self):
        """Test that base class cannot be instantiated."""
        # BaseStandardMLEstimator doesn't have __init__, so we test subclass enforcement
        with pytest.raises(TypeError, match="must define"):

            class IncompleteModel(BaseStandardMLEstimator):
                pass

    def test_subclass_requires_all_attributes(self):
        """Test that subclasses must define required attributes."""
        with pytest.raises(TypeError, match="must define"):

            class IncompleteModel(BaseStandardMLEstimator):
                default_search_space = ()

    def test_valid_subclass_with_all_attributes(self):
        """Test that valid subclass can be created."""

        class ValidModel(BaseStandardMLEstimator):
            default_search_space = ()
            _classifier_class = None
            _regressor_class = None

        # Should not raise
        assert ValidModel.default_search_space == ()


# ==============================================================================
# LGBM MODEL TESTS
# ==============================================================================


class TestLGBMModel:
    """Test LGBMModel wrapper."""

    def test_has_search_space(self):
        """Test that LGBMModel has default_search_space."""
        assert hasattr(LGBMModel, "default_search_space")
        assert LGBMModel.default_search_space is not None
        assert len(LGBMModel.default_search_space) > 0

    def test_search_space_all_valid_specs(self):
        """Test that all search space items are SearchSpaceSpec instances."""
        for spec in LGBMModel.default_search_space:
            assert isinstance(spec, SearchSpaceSpec)

    def test_has_classifier_class(self):
        """Test that classifier class is set correctly."""
        assert LGBMModel._classifier_class == LGBMClassifier

    def test_has_regressor_class(self):
        """Test that regressor class is set correctly."""
        assert LGBMModel._regressor_class == LGBMRegressor

    def test_search_space_has_expected_params(self):
        """Test that search space includes expected hyperparameters."""
        param_names = {spec.name for spec in LGBMModel.default_search_space}
        expected_params = {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "subsample_freq",
            "colsample_bytree",
            "max_bin",
            "reg_alpha",
            "reg_lambda",
            "verbosity",
        }
        assert param_names == expected_params


# ==============================================================================
# XGBOOST MODEL TESTS
# ==============================================================================


class TestXGBoostModel:
    """Test XGBoostModel wrapper."""

    def test_has_search_space(self):
        """Test that XGBoostModel has default_search_space."""
        assert hasattr(XGBoostModel, "default_search_space")
        assert len(XGBoostModel.default_search_space) > 0

    def test_has_classifier_and_regressor(self):
        """Test that both classifier and regressor classes are set."""
        assert XGBoostModel._classifier_class == XGBClassifier
        assert XGBoostModel._regressor_class == XGBRegressor

    def test_uses_lossguide_policy(self):
        """Test that XGBoost uses lossguide grow policy."""
        from caml.automl.search_space import ConstantSpec

        param_names = {spec.name for spec in XGBoostModel.default_search_space}
        assert "grow_policy" in param_names
        # Find the grow_policy spec
        grow_policy_spec = next(
            spec
            for spec in XGBoostModel.default_search_space
            if spec.name == "grow_policy"
        )
        assert isinstance(grow_policy_spec, ConstantSpec)
        assert grow_policy_spec.value == "lossguide"

    def test_search_space_has_max_leaves_not_max_depth(self):
        """Test that unlimited depth variant uses max_leaves."""
        param_names = {spec.name for spec in XGBoostModel.default_search_space}
        assert "max_leaves" in param_names
        assert "max_depth" not in param_names


class TestXGBoostLimitDepthModel:
    """Test XGBoostLimitDepthModel wrapper."""

    def test_has_search_space(self):
        """Test that XGBoostLimitDepthModel has default_search_space."""
        assert len(XGBoostLimitDepthModel.default_search_space) > 0

    def test_has_classifier_and_regressor(self):
        """Test that both classifier and regressor classes are set."""
        assert XGBoostLimitDepthModel._classifier_class == XGBClassifier
        assert XGBoostLimitDepthModel._regressor_class == XGBRegressor

    def test_search_space_has_max_depth_not_max_leaves(self):
        """Test that limited depth variant uses max_depth."""
        param_names = {
            spec.name for spec in XGBoostLimitDepthModel.default_search_space
        }
        assert "max_depth" in param_names
        assert "max_leaves" not in param_names
        assert "grow_policy" not in param_names


# ==============================================================================
# RANDOM FOREST MODEL TESTS
# ==============================================================================


class TestRandomForestModel:
    """Test RandomForestModel wrapper."""

    def test_has_search_space(self):
        """Test that RandomForestModel has default_search_space."""
        assert len(RandomForestModel.default_search_space) > 0

    def test_has_classifier_and_regressor(self):
        """Test that both classifier and regressor classes are set."""
        assert RandomForestModel._classifier_class == RandomForestClassifier
        assert RandomForestModel._regressor_class == RandomForestRegressor

    def test_search_space_has_expected_params(self):
        """Test that search space includes expected hyperparameters."""
        param_names = {spec.name for spec in RandomForestModel.default_search_space}
        expected_params = {
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_leaf",
            "min_samples_split",
        }
        assert param_names == expected_params


# ==============================================================================
# EXTRA TREES MODEL TESTS
# ==============================================================================


class TestExtraTreesModel:
    """Test ExtraTreesModel wrapper."""

    def test_has_search_space(self):
        """Test that ExtraTreesModel has default_search_space."""
        assert len(ExtraTreesModel.default_search_space) > 0

    def test_has_classifier_and_regressor(self):
        """Test that both classifier and regressor classes are set."""
        assert ExtraTreesModel._classifier_class == ExtraTreesClassifier
        assert ExtraTreesModel._regressor_class == ExtraTreesRegressor

    def test_search_space_same_as_random_forest(self):
        """Test that ExtraTrees uses same hyperparameters as RandomForest."""
        extra_trees_params = {
            spec.name for spec in ExtraTreesModel.default_search_space
        }
        random_forest_params = {
            spec.name for spec in RandomForestModel.default_search_space
        }
        assert extra_trees_params == random_forest_params


# ==============================================================================
# LOGISTIC REGRESSION MODEL TESTS
# ==============================================================================


# class TestLogisticRegressionModel:
#     """Test LogisticRegressionModel wrapper."""

#     def test_has_search_space(self):
#         """Test that LogisticRegressionModel has default_search_space."""
#         assert len(LogisticRegressionModel.default_search_space) > 0

#     def test_has_classifier_only(self):
#         """Test that only classifier class is set (no regressor)."""
#         assert LogisticRegressionModel._classifier_class == LogisticRegression
#         assert LogisticRegressionModel._regressor_class is None

#     def test_uses_saga_solver(self):
#         """Test that logistic regression uses saga solver."""
#         from caml.automl.search_space import ConstantSpec

#         param_names = {
#             spec.name for spec in LogisticRegressionModel.default_search_space
#         }
#         assert "solver" in param_names
#         solver_spec = next(
#             spec
#             for spec in LogisticRegressionModel.default_search_space
#             if spec.name == "solver"
#         )
#         assert isinstance(solver_spec, ConstantSpec)
#         assert solver_spec.value == "saga"

#     def test_search_space_has_l1_ratio(self):
#         """Test that search space includes l1_ratio for elastic net penalty."""
#         param_names = {
#             spec.name for spec in LogisticRegressionModel.default_search_space
#         }
#         assert "l1_ratio" in param_names


# ==============================================================================
# ELASTIC NET MODEL TESTS
# ==============================================================================


class TestElasticNetModel:
    """Test ElasticNetModel wrapper."""

    def test_has_search_space(self):
        """Test that ElasticNetModel has default_search_space."""
        assert len(ElasticNetModel.default_search_space) > 0

    def test_has_regressor_only(self):
        """Test that only regressor class is set (no classifier)."""
        assert ElasticNetModel._regressor_class == ElasticNet
        assert ElasticNetModel._classifier_class is None

    def test_search_space_has_expected_params(self):
        """Test that search space includes expected hyperparameters."""
        param_names = {spec.name for spec in ElasticNetModel.default_search_space}
        expected_params = {"alpha", "l1_ratio", "selection", "max_iter"}
        assert param_names == expected_params
