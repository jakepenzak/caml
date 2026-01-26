"""Tests for DR estimator wrappers."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression

from caml.data import TreatmentType
from caml.estimators import AutoCateEstimator, InferenceProvider
from caml.estimators.wrappers.dr import (
    WrappedDRLearner,
    WrappedForestDRLearner,
    WrappedLinearDRLearner,
    WrappedSparseLinearDRLearner,
)
from caml.inference import InferenceType


# All DR estimators
DR_ESTIMATORS = [
    WrappedDRLearner,
    WrappedLinearDRLearner,
    WrappedSparseLinearDRLearner,
    WrappedForestDRLearner,
]


@pytest.mark.parametrize("estimator_class", DR_ESTIMATORS)
class TestDRWrappers:
    """Common tests for all DR wrappers."""

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
        # All DR estimators should support binary treatment + continuous outcome
        assert result is True

    def test_protocol_compliance(self, estimator_class):
        """Test estimator implements AutoCateEstimator."""
        estimator = estimator_class()
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_basic(self, estimator_class, small_binary_continuous_data):
        """Test basic fit functionality."""
        estimator = estimator_class(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )
        result = estimator.fit(small_binary_continuous_data)
        assert result is estimator  # Returns self
        assert estimator._is_fitted is True

    def test_fit_sets_fitted_flag(self, estimator_class, small_binary_continuous_data):
        """Test _is_fitted is set after fit."""
        estimator = estimator_class(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
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
        estimator = estimator_class(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)

        effects = estimator.effect(small_binary_continuous_data.X)
        assert isinstance(effects, np.ndarray)
        assert effects.shape[0] == len(small_binary_continuous_data.X)

    def test_does_not_support_continuous_treatment(self, estimator_class):
        """Test DR estimators don't support continuous treatment."""
        estimator = estimator_class()
        assert TreatmentType.CONTINUOUS not in estimator.capabilities.treatment_types


class TestDRLearner:
    """Specific tests for WrappedDRLearner."""

    def test_supports_bootstrap_inference_only(self):
        """Test DRLearner supports bootstrap inference only."""
        estimator = WrappedDRLearner()
        assert InferenceType.BOOTSTRAP in estimator.capabilities.inference_types
        assert InferenceType.ANALYTIC not in estimator.capabilities.inference_types

    def test_flexible_model_specification(self, small_binary_continuous_data):
        """Test DRLearner with flexible model choices."""
        estimator = WrappedDRLearner(
            model_propensity=RandomForestClassifier(n_estimators=20),
            model_regression=RandomForestRegressor(n_estimators=20),
            model_final=RandomForestRegressor(n_estimators=20),
            cv=2,
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)


class TestLinearDRLearner:
    """Specific tests for WrappedLinearDRLearner."""

    def test_supports_analytic_inference(self):
        """Test LinearDRLearner supports analytic inference."""
        estimator = WrappedLinearDRLearner()
        assert InferenceType.ANALYTIC in estimator.capabilities.inference_types
        assert estimator.capabilities.supports_inference is True

    def test_inference_basic(self, small_binary_continuous_data):
        """Test effect_inference returns InferenceResult."""
        estimator = WrappedLinearDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)

        # Should be InferenceProvider
        assert isinstance(estimator, InferenceProvider)

        result = estimator.effect_inference(small_binary_continuous_data.X)
        assert result.effect is not None
        assert result.stderr is not None
        assert result.effect.shape[0] == len(small_binary_continuous_data.X)


class TestSparseLinearDRLearner:
    """Specific tests for WrappedSparseLinearDRLearner."""

    def test_supports_analytic_inference(self):
        """Test SparseLinearDRLearner supports analytic inference."""
        estimator = WrappedSparseLinearDRLearner()
        assert InferenceType.ANALYTIC in estimator.capabilities.inference_types

    def test_with_high_dimensional_data(self, high_dimensional_data):
        """Test SparseLinearDRLearner with many features (sparsity)."""
        estimator = WrappedSparseLinearDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            alpha=0.1,
            cv=2,
            random_state=42,
        )
        estimator.fit(high_dimensional_data)
        effects = estimator.effect(high_dimensional_data.X)
        assert effects.shape[0] == len(high_dimensional_data.X)


class TestForestDRLearner:
    """Specific tests for WrappedForestDRLearner."""

    def test_supports_bootstrap_inference_only(self):
        """Test ForestDRLearner supports bootstrap inference only."""
        estimator = WrappedForestDRLearner()
        assert InferenceType.BOOTSTRAP in estimator.capabilities.inference_types
        assert InferenceType.ANALYTIC not in estimator.capabilities.inference_types

    def test_with_forest_parameters(self, small_binary_continuous_data):
        """Test ForestDRLearner with forest-specific parameters."""
        estimator = WrappedForestDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=10,
            cv=2,
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)


class TestDREdgeCases:
    """Edge case tests for DR estimators."""

    def test_fit_with_binary_outcome(self, binary_binary_data):
        """Test DR estimators handle binary outcomes."""
        estimator = WrappedLinearDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )
        assert WrappedLinearDRLearner.is_compatible_with(binary_binary_data)
        estimator.fit(binary_binary_data)
        effects = estimator.effect(binary_binary_data.X)
        assert effects.shape[0] == len(binary_binary_data.X)

    def test_discrete_outcome_flag_set_correctly(self, binary_continuous_data):
        """Test discrete_outcome flag is set correctly."""
        estimator = WrappedDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        # Check flag was set
        assert hasattr(estimator._estimator, "discrete_outcome")
        assert estimator._estimator.discrete_outcome is False

    def test_with_confounders_W(self, binary_continuous_data):
        """Test DR estimators handle confounders (W)."""
        # Add some confounders to the data
        W = np.random.randn(len(binary_continuous_data.X), 2)
        binary_continuous_data._W = W

        estimator = WrappedLinearDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )
        estimator.fit(binary_continuous_data)
        effects = estimator.effect(binary_continuous_data.X)
        assert effects.shape[0] == len(binary_continuous_data.X)

    def test_with_sample_weights(self, small_binary_continuous_data):
        """Test DR estimators handle sample weights."""
        # Add weights
        weights = np.random.rand(len(small_binary_continuous_data.X))
        small_binary_continuous_data._weights = weights

        estimator = WrappedLinearDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )

        # Should handle weights
        assert estimator.capabilities.supports_weights is True
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)

    def test_multi_valued_treatment(self, multi_continuous_data):
        """Test DR estimators support multi-valued treatment."""
        estimator = WrappedLinearDRLearner(
            model_propensity=LogisticRegression(max_iter=500),
            model_regression=LassoCV(),
            cv=2,
            random_state=42,
        )

        # All DR estimators support multi-valued treatment
        assert WrappedLinearDRLearner.is_compatible_with(multi_continuous_data)
        estimator.fit(multi_continuous_data)
        effects = estimator.effect(multi_continuous_data.X)
        assert effects.shape[0] == len(multi_continuous_data.X)
