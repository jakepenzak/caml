"""Tests for Orthogonal Random Forest estimator wrappers."""

import numpy as np
import pytest
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from caml.estimators import AutoCateEstimator
from caml.estimators.wrappers.orf import WrappedDMLOrthoForest, WrappedDROrthoForest
from caml.inference import InferenceType

# All ORF estimators
ORF_ESTIMATORS = [
    WrappedDMLOrthoForest,
    WrappedDROrthoForest,
]


@pytest.mark.parametrize("estimator_class", ORF_ESTIMATORS)
class TestORFWrappers:
    """Common tests for all ORF wrappers."""

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
        # All ORF estimators should support binary treatment + continuous outcome
        assert result is True

    def test_protocol_compliance(self, estimator_class):
        """Test estimator implements AutoCateEstimator."""
        estimator = estimator_class()
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_basic(self, estimator_class, small_binary_continuous_data):
        """Test basic fit functionality."""
        # Different ORF estimators need different model specs
        if estimator_class == WrappedDMLOrthoForest:
            estimator = estimator_class(
                n_trees=50,
                max_depth=20,
                min_leaf_size=5,
                model_T=LogisticRegressionCV(),
                model_Y=LassoCV(),
                random_state=42,
            )
        elif estimator_class == WrappedDROrthoForest:
            estimator = estimator_class(
                n_trees=50,
                max_depth=20,
                min_leaf_size=5,
                propensity_model=LogisticRegressionCV(),
                model_Y=LassoCV(),
                random_state=42,
            )

        result = estimator.fit(small_binary_continuous_data)
        assert result is estimator  # Returns self
        assert estimator._is_fitted is True

    def test_fit_sets_fitted_flag(self, estimator_class, small_binary_continuous_data):
        """Test _is_fitted is set after fit."""
        if estimator_class == WrappedDMLOrthoForest:
            estimator = estimator_class(
                n_trees=50,
                model_T=LogisticRegressionCV(),
                model_Y=LassoCV(),
                random_state=42,
            )
        elif estimator_class == WrappedDROrthoForest:
            estimator = estimator_class(
                n_trees=50,
                propensity_model=LogisticRegressionCV(),
                model_Y=LassoCV(),
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
        if estimator_class == WrappedDMLOrthoForest:
            estimator = estimator_class(
                n_trees=50,
                model_T=LogisticRegressionCV(),
                model_Y=LassoCV(),
                random_state=42,
            )
        elif estimator_class == WrappedDROrthoForest:
            estimator = estimator_class(
                n_trees=50,
                propensity_model=LogisticRegressionCV(),
                model_Y=LassoCV(),
                random_state=42,
            )

        estimator.fit(small_binary_continuous_data)

        effects = estimator.effect(small_binary_continuous_data.X)
        assert isinstance(effects, np.ndarray)
        assert effects.shape[0] == len(small_binary_continuous_data.X)

    def test_supports_bootstrap_inference_only(self, estimator_class):
        """Test ORF estimators support bootstrap inference only."""
        estimator = estimator_class()
        assert InferenceType.BOOTSTRAP in estimator.capabilities.inference_types
        assert InferenceType.ANALYTIC not in estimator.capabilities.inference_types


class TestDMLOrthoForest:
    """Specific tests for WrappedDMLOrthoForest."""

    def test_discrete_treatment_flag_handling(self, binary_continuous_data):
        """Test DMLOrthoForest sets discrete_treatment flag correctly."""
        estimator = WrappedDMLOrthoForest(
            n_trees=50,
            model_T=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        # Check flag was set
        assert hasattr(estimator._estimator, "discrete_treatment")
        assert estimator._estimator.discrete_treatment is True

    def test_supports_continuous_treatment(self, continuous_continuous_data):
        """Test DMLOrthoForest handles continuous treatment."""
        estimator = WrappedDMLOrthoForest(
            n_trees=50,
            model_T=LassoCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        assert WrappedDMLOrthoForest.is_compatible_with(continuous_continuous_data)
        estimator.fit(continuous_continuous_data)
        effects = estimator.effect(continuous_continuous_data.X)
        assert effects.shape[0] == len(continuous_continuous_data.X)

    def test_with_forest_parameters(self, small_binary_continuous_data):
        """Test DMLOrthoForest with forest-specific parameters."""
        estimator = WrappedDMLOrthoForest(
            n_trees=100,
            max_depth=50,
            min_leaf_size=10,
            subsample_ratio=0.7,
            model_T=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)


class TestDROrthoForest:
    """Specific tests for WrappedDROrthoForest."""

    def test_no_discrete_treatment_flag(self, binary_continuous_data):
        """Test DROrthoForest doesn't set discrete_treatment (different API)."""
        estimator = WrappedDROrthoForest(
            n_trees=50,
            propensity_model=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        # DROrthoForest doesn't set discrete_treatment flag
        effects = estimator.effect(binary_continuous_data.X)
        assert effects.shape[0] == len(binary_continuous_data.X)

    def test_with_doubly_robust_parameters(self, small_binary_continuous_data):
        """Test DROrthoForest with DR-specific parameters."""
        estimator = WrappedDROrthoForest(
            n_trees=100,
            max_depth=50,
            min_leaf_size=10,
            subsample_ratio=0.7,
            propensity_model=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)


class TestORFEdgeCases:
    """Edge case tests for ORF estimators."""

    def test_fit_with_binary_outcome(self, binary_binary_data):
        """Test ORF estimators handle binary outcomes."""
        estimator = WrappedDMLOrthoForest(
            n_trees=50,
            model_T=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        assert WrappedDMLOrthoForest.is_compatible_with(binary_binary_data)
        estimator.fit(binary_binary_data)
        effects = estimator.effect(binary_binary_data.X)
        assert effects.shape[0] == len(binary_binary_data.X)

    def test_with_confounders_W(self, binary_continuous_data):
        """Test ORF estimators handle confounders (W)."""
        # Add some confounders to the data
        W = np.random.randn(len(binary_continuous_data.X), 2)
        binary_continuous_data._W = W

        estimator = WrappedDMLOrthoForest(
            n_trees=50,
            model_T=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)
        effects = estimator.effect(binary_continuous_data.X)
        assert effects.shape[0] == len(binary_continuous_data.X)

    def test_does_not_support_weights(self):
        """Test ORF estimators don't support sample weights."""
        estimator = WrappedDMLOrthoForest()
        assert estimator.capabilities.supports_weights is False
