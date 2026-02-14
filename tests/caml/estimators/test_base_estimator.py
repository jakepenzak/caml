"""Tests for caml.estimators.base module."""

import numpy as np
import pytest

from caml.automl import SearchSpace
from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators import AutoCateEstimator, EstimatorCapabilities
from caml.inference import InferenceType

pytestmark = pytest.mark.estimators


# ==============================================================================
# ESTIMATOR CAPABILITIES TESTS
# ==============================================================================


class TestEstimatorCapabilities:
    """Test EstimatorCapabilities dataclass."""

    def test_create_capabilities(self):
        """Test creating EstimatorCapabilities."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.ANALYTIC},
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=True,
            supports_weights=True,
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            supports_inference=True,
        )

        assert TreatmentType.BINARY in capabilities.treatment_types
        assert OutcomeType.CONTINUOUS in capabilities.outcome_types
        assert InferenceType.ANALYTIC in capabilities.inference_types
        assert Estimand.CATE in capabilities.estimands
        assert capabilities.supports_controls_in_first_stage_only is True
        assert capabilities.supports_weights is True
        assert capabilities.requires_treatment_model is True
        assert capabilities.requires_outcome_model is True
        assert capabilities.requires_regression_model is False
        assert capabilities.supports_inference is True

    def test_capabilities_are_immutable(self):
        """Test that capabilities are frozen/immutable."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            supports_inference=False,
        )

        # Should raise since dataclass is frozen
        with pytest.raises(Exception):  # FrozenInstanceError
            capabilities.supports_weights = True

    def test_multiple_treatment_types(self):
        """Test capabilities with multiple treatment types."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.MULTI},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        assert len(capabilities.treatment_types) == 2
        assert TreatmentType.BINARY in capabilities.treatment_types
        assert TreatmentType.MULTI in capabilities.treatment_types

    def test_multiple_outcome_types(self):
        """Test capabilities with multiple outcome types."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.BINARY, OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        assert len(capabilities.outcome_types) == 2
        assert OutcomeType.BINARY in capabilities.outcome_types
        assert OutcomeType.CONTINUOUS in capabilities.outcome_types

    def test_multiple_inference_types(self):
        """Test capabilities with multiple inference types."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.ANALYTIC, InferenceType.BOOTSTRAP},
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=True,
        )

        assert len(capabilities.inference_types) == 2
        assert InferenceType.ANALYTIC in capabilities.inference_types
        assert InferenceType.BOOTSTRAP in capabilities.inference_types

    def test_empty_inference_types(self):
        """Test capabilities with no inference support."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        assert len(capabilities.inference_types) == 0


# ==============================================================================
# COMPATIBILITY CHECKING TESTS
# ==============================================================================


class TestCompatibilityChecking:
    """Test EstimatorCapabilities.is_compatible() method."""

    def test_compatible_binary_continuous(self):
        """Test compatibility with binary treatment and continuous outcome."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            supports_inference=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert capabilities.is_compatible(data) is True

    def test_incompatible_treatment_type(self):
        """Test incompatibility due to treatment type."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.randn(100),  # Continuous
            Y=np.random.randn(100),
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert capabilities.is_compatible(data) is False

    def test_incompatible_outcome_type(self):
        """Test incompatibility due to outcome type."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.binomial(1, 0.5, 100),  # Binary
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.BINARY,
        )

        assert capabilities.is_compatible(data) is False

    def test_compatible_multiple_types(self):
        """Test compatibility when estimator supports multiple types."""
        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.BINARY, OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        # Test binary treatment, continuous outcome
        np.random.seed(42)
        data1 = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )
        assert capabilities.is_compatible(data1) is True

        # Test continuous treatment, binary outcome
        data2 = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.randn(100),
            Y=np.random.binomial(1, 0.5, 100),
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.BINARY,
        )
        assert capabilities.is_compatible(data2) is True


# ==============================================================================
# PROTOCOL TESTS
# ==============================================================================


class TestAutoCateEstimatorProtocol:
    """Test AutoCateEstimator protocol."""

    def test_simple_estimator_implements_protocol(self):
        """Test that a simple estimator implements the protocol."""

        class SimpleEstimator:
            capabilities: EstimatorCapabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types=set(),
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                supports_inference=False,
            )

            default_search_space: SearchSpace = ()

            def __init__(self):
                self.effect_value = None

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                temp_instance = cls()
                return temp_instance.capabilities.is_compatible(data)

            def fit(self, data: CausalDataset, **kwargs):
                T = np.asarray(data.T)
                Y = np.asarray(data.Y)
                self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
                return self

            def effect(self, X, **kwargs):
                n = len(X) if hasattr(X, "__len__") else 1
                return np.full(n, self.effect_value)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        estimator = SimpleEstimator()
        assert isinstance(estimator, AutoCateEstimator)

    def test_incomplete_estimator_does_not_implement_protocol(self):
        """Test that incomplete estimator doesn't implement protocol."""

        class IncompleteEstimator:
            pass

        estimator = IncompleteEstimator()
        assert not isinstance(estimator, AutoCateEstimator)
