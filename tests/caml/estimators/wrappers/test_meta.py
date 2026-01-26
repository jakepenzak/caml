"""Tests for Meta-learner estimator wrappers."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV

from caml.estimators import AutoCateEstimator
from caml.estimators.wrappers.meta import (
    WrappedSLearner,
    WrappedTLearner,
    WrappedXLearner,
)
from caml.inference import InferenceType

# All meta-learner estimators
META_ESTIMATORS = [
    WrappedSLearner,
    WrappedTLearner,
    WrappedXLearner,
]


@pytest.mark.parametrize("estimator_class", META_ESTIMATORS)
class TestMetaLearnerWrappers:
    """Common tests for all meta-learner wrappers."""

    def test_initialization(self, estimator_class):
        """Test estimator can be instantiated."""
        estimator = estimator_class()
        assert estimator is not None
        assert estimator._is_fitted is False

    def test_capabilities_property(self, estimator_class):
        """Test capabilities returns EstimatorCapabilities."""
        estimator = estimator_class()
        caps = estimator.capabilities
        assert caps is not None
        assert hasattr(caps, "treatment_types")
        assert hasattr(caps, "outcome_types")

    def test_clean_name_property(self, estimator_class):
        """Test clean_name returns string."""
        estimator = estimator_class()
        assert isinstance(estimator.clean_name, str)
        assert len(estimator.clean_name) > 0

    def test_is_compatible_with_classmethod(
        self, estimator_class, binary_continuous_data
    ):
        """Test class-level compatibility check."""
        result = estimator_class.is_compatible_with(binary_continuous_data)
        assert isinstance(result, bool)
        # All meta-learners should support binary treatment + continuous outcome
        assert result is True

    def test_protocol_compliance(self, estimator_class):
        """Test estimator implements AutoCateEstimator."""
        estimator = estimator_class()
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_basic(self, estimator_class, small_binary_continuous_data):
        """Test basic fit functionality."""
        # Different meta-learners need different model specs
        if estimator_class == WrappedSLearner:
            estimator = estimator_class(
                overall_model=GradientBoostingRegressor(n_estimators=20),
            )
        elif estimator_class == WrappedTLearner:
            estimator = estimator_class(
                models=GradientBoostingRegressor(n_estimators=20),
            )
        elif estimator_class == WrappedXLearner:
            estimator = estimator_class(
                models=GradientBoostingRegressor(n_estimators=20),
                cate_models=GradientBoostingRegressor(n_estimators=20),
                propensity_model=LogisticRegressionCV(),
            )

        result = estimator.fit(small_binary_continuous_data)
        assert result is estimator  # Returns self
        assert estimator._is_fitted is True

    def test_fit_sets_fitted_flag(self, estimator_class, small_binary_continuous_data):
        """Test _is_fitted is set after fit."""
        # Different meta-learners need different model specs
        if estimator_class == WrappedSLearner:
            estimator = estimator_class(
                overall_model=GradientBoostingRegressor(n_estimators=20),
            )
        elif estimator_class == WrappedTLearner:
            estimator = estimator_class(
                models=GradientBoostingRegressor(n_estimators=20),
            )
        elif estimator_class == WrappedXLearner:
            estimator = estimator_class(
                models=GradientBoostingRegressor(n_estimators=20),
                cate_models=GradientBoostingRegressor(n_estimators=20),
                propensity_model=LogisticRegressionCV(),
            )

        assert estimator._is_fitted is False
        estimator.fit(small_binary_continuous_data)
        assert estimator._is_fitted is True

    def test_effect_before_fit_raises(
        self, estimator_class, small_binary_continuous_data
    ):
        """Test calling effect before fit raises error."""
        estimator = estimator_class()
        with pytest.raises(RuntimeError, match="must be fitted"):
            estimator.effect(small_binary_continuous_data.X)

    def test_effect_after_fit(self, estimator_class, small_binary_continuous_data):
        """Test effect returns array of correct shape."""
        # Different meta-learners need different model specs
        if estimator_class == WrappedSLearner:
            estimator = estimator_class(
                overall_model=GradientBoostingRegressor(n_estimators=20),
            )
        elif estimator_class == WrappedTLearner:
            estimator = estimator_class(
                models=GradientBoostingRegressor(n_estimators=20),
            )
        elif estimator_class == WrappedXLearner:
            estimator = estimator_class(
                models=GradientBoostingRegressor(n_estimators=20),
                cate_models=GradientBoostingRegressor(n_estimators=20),
                propensity_model=LogisticRegressionCV(),
            )

        estimator.fit(small_binary_continuous_data)

        effects = estimator.effect(small_binary_continuous_data.X)
        assert isinstance(effects, np.ndarray)
        assert effects.shape[0] == len(small_binary_continuous_data.X)

    def test_supports_bootstrap_inference_only(self, estimator_class):
        """Test meta-learners support bootstrap inference only."""
        estimator = estimator_class()
        assert InferenceType.BOOTSTRAP in estimator.capabilities.inference_types
        assert InferenceType.ANALYTIC not in estimator.capabilities.inference_types


class TestSLearner:
    """Specific tests for WrappedSLearner."""

    def test_single_model_approach(self, small_binary_continuous_data):
        """Test S-Learner uses single model."""
        estimator = WrappedSLearner(
            overall_model=RandomForestRegressor(n_estimators=30),
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)

    def test_does_not_support_weights(self):
        """Test S-Learner doesn't support sample weights."""
        estimator = WrappedSLearner()
        assert estimator.capabilities.supports_weights is False

    def test_requires_regression_model(self):
        """Test S-Learner requires regression model."""
        estimator = WrappedSLearner()
        assert estimator.capabilities.requires_regression_model is True
        assert estimator.capabilities.requires_treatment_model is False
        assert estimator.capabilities.requires_outcome_model is False


class TestTLearner:
    """Specific tests for WrappedTLearner."""

    def test_separate_models_per_treatment(self, small_binary_continuous_data):
        """Test T-Learner trains separate models per treatment."""
        estimator = WrappedTLearner(
            models=RandomForestRegressor(n_estimators=30),
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)

    def test_does_not_support_weights(self):
        """Test T-Learner doesn't support sample weights."""
        estimator = WrappedTLearner()
        assert estimator.capabilities.supports_weights is False


class TestXLearner:
    """Specific tests for WrappedXLearner."""

    def test_requires_propensity_model(self):
        """Test X-Learner requires propensity model."""
        estimator = WrappedXLearner()
        assert estimator.capabilities.requires_treatment_model is True

    def test_with_propensity_weighting(self, small_binary_continuous_data):
        """Test X-Learner uses propensity score weighting."""
        estimator = WrappedXLearner(
            models=GradientBoostingRegressor(n_estimators=20),
            cate_models=GradientBoostingRegressor(n_estimators=20),
            propensity_model=LogisticRegressionCV(),
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)


class TestMetaLearnerEdgeCases:
    """Edge case tests for meta-learner estimators."""

    def test_fit_with_binary_outcome(self, binary_binary_data):
        """Test meta-learners handle binary outcomes."""
        estimator = WrappedSLearner(
            overall_model=GradientBoostingRegressor(n_estimators=20),
        )
        assert WrappedSLearner.is_compatible_with(binary_binary_data)
        estimator.fit(binary_binary_data)
        effects = estimator.effect(binary_binary_data.X)
        assert effects.shape[0] == len(binary_binary_data.X)

    def test_multi_valued_treatment(self, multi_continuous_data):
        """Test meta-learners support multi-valued treatment."""
        estimator = WrappedSLearner(
            overall_model=GradientBoostingRegressor(n_estimators=20),
        )

        # All meta-learners support multi-valued treatment
        assert WrappedSLearner.is_compatible_with(multi_continuous_data)
        estimator.fit(multi_continuous_data)
        effects = estimator.effect(multi_continuous_data.X)
        assert effects.shape[0] == len(multi_continuous_data.X)

    def test_no_controls_in_first_stage_only(self):
        """Test meta-learners don't support controls in first stage only."""
        estimator = WrappedSLearner()
        assert estimator.capabilities.supports_controls_in_first_stage_only is False
