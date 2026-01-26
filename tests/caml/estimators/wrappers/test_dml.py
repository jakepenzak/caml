"""Tests for DML estimator wrappers."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from caml.data import TreatmentType
from caml.estimators import AutoCateEstimator, InferenceProvider
from caml.estimators.wrappers.dml import (
    WrappedCausalForestDML,
    WrappedKernelDML,
    WrappedLinearDML,
    WrappedNonParamDML,
    WrappedSparseLinearDML,
)
from caml.inference import InferenceType

# All DML estimators
DML_ESTIMATORS = [
    WrappedLinearDML,
    WrappedSparseLinearDML,
    WrappedCausalForestDML,
    WrappedNonParamDML,
    WrappedKernelDML,
]


@pytest.mark.parametrize("estimator_class", DML_ESTIMATORS)
class TestDMLWrappers:
    """Common tests for all DML wrappers."""

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
        # All DML estimators should support binary treatment + continuous outcome
        assert result is True

    def test_protocol_compliance(self, estimator_class):
        """Test estimator implements AutoCateEstimator."""
        estimator = estimator_class()
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_basic(self, estimator_class, small_binary_continuous_data):
        """Test basic fit functionality."""
        estimator = estimator_class(
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
            cv=2,
            random_state=42,
        )
        result = estimator.fit(small_binary_continuous_data)
        assert result is estimator  # Returns self
        assert estimator._is_fitted is True

    def test_fit_sets_fitted_flag(self, estimator_class, small_binary_continuous_data):
        """Test _is_fitted is set after fit."""
        estimator = estimator_class(
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
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
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
            cv=2,
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)

        effects = estimator.effect(small_binary_continuous_data.X)
        assert isinstance(effects, np.ndarray)
        assert effects.shape[0] == len(small_binary_continuous_data.X)

    def test_check_compatibility_raises_on_incompatible(
        self, estimator_class, multi_continuous_data
    ):
        """Test check_compatibility raises detailed error for incompatible data."""
        # NonParamDML doesn't support multi-valued treatment
        if estimator_class == WrappedNonParamDML:
            estimator = estimator_class()
            with pytest.raises(ValueError, match="Data incompatible"):
                estimator.check_compatibility(multi_continuous_data, raise_error=True)


class TestLinearDML:
    """Specific tests for WrappedLinearDML."""

    def test_supports_analytic_inference(self):
        """Test LinearDML supports analytic inference."""
        estimator = WrappedLinearDML()
        assert InferenceType.ANALYTIC in estimator.capabilities.inference_types
        assert estimator.capabilities.supports_inference is True

    def test_inference_basic(self, small_binary_continuous_data):
        """Test effect_inference returns InferenceResult."""
        estimator = WrappedLinearDML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
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

    def test_supports_continuous_treatment(self, continuous_continuous_data):
        """Test LinearDML handles continuous treatment."""
        assert WrappedLinearDML.is_compatible_with(continuous_continuous_data)

        estimator = WrappedLinearDML(
            model_y=LinearRegression(),
            model_t=LinearRegression(),
            cv=2,
            random_state=42,
        )
        estimator.fit(continuous_continuous_data)
        effects = estimator.effect(continuous_continuous_data.X)
        assert effects.shape[0] == len(continuous_continuous_data.X)


class TestSparseLinearDML:
    """Specific tests for WrappedSparseLinearDML."""

    def test_supports_analytic_inference(self):
        """Test SparseLinearDML supports analytic inference."""
        estimator = WrappedSparseLinearDML()
        assert InferenceType.ANALYTIC in estimator.capabilities.inference_types

    def test_with_high_dimensional_data(self, high_dimensional_data):
        """Test SparseLinearDML with many features (sparsity)."""
        estimator = WrappedSparseLinearDML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
            alpha=0.1,
            cv=2,
            random_state=42,
        )
        estimator.fit(high_dimensional_data)
        effects = estimator.effect(high_dimensional_data.X)
        assert effects.shape[0] == len(high_dimensional_data.X)


class TestCausalForestDML:
    """Specific tests for WrappedCausalForestDML."""

    def test_supports_bootstrap_inference_only(self):
        """Test CausalForestDML supports bootstrap inference."""
        estimator = WrappedCausalForestDML()
        assert InferenceType.BOOTSTRAP in estimator.capabilities.inference_types
        assert InferenceType.ANALYTIC not in estimator.capabilities.inference_types

    def test_with_nonlinear_data(self, binary_continuous_data):
        """Test CausalForestDML with nonlinear effects."""
        estimator = WrappedCausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=20),
            model_t=GradientBoostingClassifier(n_estimators=20),
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=10,
            cv=2,
            random_state=42,
        )
        estimator.fit(binary_continuous_data)
        effects = estimator.effect(binary_continuous_data.X)
        assert effects.shape[0] == len(binary_continuous_data.X)


class TestNonParamDML:
    """Specific tests for WrappedNonParamDML."""

    def test_does_not_support_multi_treatment(self):
        """Test NonParamDML doesn't support multi-valued treatment."""
        estimator = WrappedNonParamDML()
        assert TreatmentType.MULTI not in estimator.capabilities.treatment_types

    def test_supports_bootstrap_only(self):
        """Test NonParamDML supports bootstrap inference only."""
        estimator = WrappedNonParamDML()
        assert InferenceType.BOOTSTRAP in estimator.capabilities.inference_types
        assert InferenceType.ANALYTIC not in estimator.capabilities.inference_types


class TestKernelDML:
    """Specific tests for WrappedKernelDML."""

    def test_supports_analytic_inference(self):
        """Test KernelDML supports analytic inference."""
        estimator = WrappedKernelDML()
        assert InferenceType.ANALYTIC in estimator.capabilities.inference_types

    def test_with_kernel_parameters(self, small_binary_continuous_data):
        """Test KernelDML accepts kernel-specific parameters."""
        # Test that kernel parameters are passed through to EconML
        estimator = WrappedKernelDML(
            model_y=GradientBoostingRegressor(n_estimators=20),
            model_t=GradientBoostingClassifier(n_estimators=20),
            cv=2,
            random_state=42,
        )
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)


class TestDMLEdgeCases:
    """Edge case tests for DML estimators."""

    def test_fit_with_binary_outcome(self, binary_binary_data):
        """Test DML estimators handle binary outcomes."""
        # All DML estimators should support binary outcomes
        estimator = WrappedLinearDML(
            model_y=LogisticRegression(max_iter=500),
            model_t=LogisticRegression(max_iter=500),
            cv=2,
            random_state=42,
        )
        assert WrappedLinearDML.is_compatible_with(binary_binary_data)
        estimator.fit(binary_binary_data)
        effects = estimator.effect(binary_binary_data.X)
        assert effects.shape[0] == len(binary_binary_data.X)

    def test_discrete_flags_set_correctly(self, binary_continuous_data):
        """Test discrete_treatment and discrete_outcome flags are set correctly."""
        estimator = WrappedLinearDML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
            cv=2,
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        # Check flags were set
        assert hasattr(estimator._estimator, "discrete_treatment")
        assert hasattr(estimator._estimator, "discrete_outcome")
        assert estimator._estimator.discrete_treatment is True
        assert estimator._estimator.discrete_outcome is False

    def test_with_confounders_W(self, binary_continuous_data):
        """Test DML estimators handle confounders (W)."""
        # Add some confounders to the data
        W = np.random.randn(len(binary_continuous_data.X), 2)
        binary_continuous_data._W = W

        estimator = WrappedLinearDML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
            cv=2,
            random_state=42,
        )
        estimator.fit(binary_continuous_data)
        effects = estimator.effect(binary_continuous_data.X)
        assert effects.shape[0] == len(binary_continuous_data.X)

    def test_with_sample_weights(self, small_binary_continuous_data):
        """Test DML estimators handle sample weights."""
        # Add weights
        weights = np.random.rand(len(small_binary_continuous_data.X))
        small_binary_continuous_data._weights = weights

        estimator = WrappedLinearDML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(max_iter=500),
            cv=2,
            random_state=42,
        )

        # Should handle weights (all DML support weights)
        assert estimator.capabilities.supports_weights is True
        estimator.fit(small_binary_continuous_data)
        effects = estimator.effect(small_binary_continuous_data.X)
        assert effects.shape[0] == len(small_binary_continuous_data.X)
