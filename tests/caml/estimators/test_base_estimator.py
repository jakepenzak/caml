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


# ==============================================================================
# BASE AUTO CATE ESTIMATOR MIXIN TESTS
# ==============================================================================


class TestBaseAutoCateEstimatorMixin:
    """Test BaseAutoCateEstimatorMixin abstract base class."""

    def test_requires_capabilities_attribute(self):
        """Test that subclass must define capabilities."""
        from caml.automl import IntSpec
        from caml.estimators import BaseAutoCateEstimatorMixin

        with pytest.raises(TypeError, match="must define 'capabilities'"):

            class MissingCapabilities(BaseAutoCateEstimatorMixin):
                default_search_space = (IntSpec(name="x", lower=1, upper=10),)

                def fit(self, data, **kwargs):
                    return self

                def effect(self, X, **kwargs):
                    return np.zeros(len(X))

    def test_requires_default_search_space_attribute(self):
        """Test that subclass must define default_search_space."""
        from caml.estimators import BaseAutoCateEstimatorMixin

        with pytest.raises(TypeError, match="must define 'default_search_space'"):

            class MissingSearchSpace(BaseAutoCateEstimatorMixin):
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

                def fit(self, data, **kwargs):
                    return self

                def effect(self, X, **kwargs):
                    return np.zeros(len(X))

    def test_valid_subclass_can_be_created(self):
        """Test that valid subclass with all required attrs can be created."""
        from caml.automl import IntSpec
        from caml.estimators import BaseAutoCateEstimatorMixin

        class ValidEstimator(BaseAutoCateEstimatorMixin):
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

            default_search_space = (IntSpec(name="param", lower=1, upper=10),)

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

            def effect(self, X, **kwargs):
                return np.zeros(len(X))

        # Should not raise
        est = ValidEstimator()
        assert hasattr(est, "capabilities")
        assert hasattr(est, "default_search_space")

    def test_is_compatible_with_uses_capabilities(self):
        """Test that is_compatible_with delegates to capabilities."""
        from caml.automl import IntSpec
        from caml.estimators import BaseAutoCateEstimatorMixin

        class TestEstimator(BaseAutoCateEstimatorMixin):
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

            default_search_space = (IntSpec(name="param", lower=1, upper=10),)

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

            def effect(self, X, **kwargs):
                return np.zeros(len(X))

        # Create compatible data
        np.random.seed(42)
        compatible_data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        # Create incompatible data
        incompatible_data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.randn(100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert TestEstimator.is_compatible_with(compatible_data) is True
        assert TestEstimator.is_compatible_with(incompatible_data) is False

    def test_check_fitted_raises_before_fit(self):
        """Test that check_fitted raises error if not fitted."""
        from caml.automl import IntSpec
        from caml.estimators import BaseAutoCateEstimatorMixin

        class TestEstimator(BaseAutoCateEstimatorMixin):
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

            default_search_space = (IntSpec(name="param", lower=1, upper=10),)

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

            def effect(self, X, **kwargs):
                self.check_fitted()
                return np.zeros(len(X))

        est = TestEstimator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            est.effect(np.random.randn(10, 3))

    def test_check_fitted_passes_after_fit(self):
        """Test that check_fitted passes after fit."""
        from caml.automl import IntSpec
        from caml.estimators import BaseAutoCateEstimatorMixin

        class TestEstimator(BaseAutoCateEstimatorMixin):
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

            default_search_space = (IntSpec(name="param", lower=1, upper=10),)

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

            def effect(self, X, **kwargs):
                self.check_fitted()
                return np.zeros(len(X))

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        est = TestEstimator()
        est.fit(data)
        # Should not raise
        result = est.effect(np.random.randn(10, 3))
        assert result.shape == (10,)

    def test_inherits_sklearn_base_estimator(self):
        """Test that BaseAutoCateEstimatorMixin inherits from sklearn.base.BaseEstimator."""
        from sklearn.base import BaseEstimator

        from caml.estimators import BaseAutoCateEstimatorMixin

        assert issubclass(BaseAutoCateEstimatorMixin, BaseEstimator)

    def test_get_params_and_set_params_from_sklearn(self):
        """Test that get_params and set_params work (inherited from sklearn)."""
        from caml.automl import IntSpec
        from caml.estimators import BaseAutoCateEstimatorMixin

        class TestEstimator(BaseAutoCateEstimatorMixin):
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

            default_search_space = (IntSpec(name="param", lower=1, upper=10),)

            def __init__(self, param=5):
                self.param = param

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

            def effect(self, X, **kwargs):
                return np.zeros(len(X))

        est = TestEstimator(param=7)
        params = est.get_params()
        assert "param" in params
        assert params["param"] == 7

        est.set_params(param=10)
        assert est.param == 10


# ==============================================================================
# INFERENCE PROVIDER PROTOCOL TESTS
# ==============================================================================


class TestInferenceProviderProtocol:
    """Test InferenceProvider protocol."""

    def test_estimator_with_inference_implements_protocol(self):
        """Test that estimator with effect_inference implements InferenceProvider."""
        from caml.estimators import InferenceProvider
        from caml.inference import InferenceResult

        class EstimatorWithInference:
            def effect_inference(
                self,
                X,
                inference_type=None,
                bootstrapper=None,
                **effect_inference_kwargs,
            ) -> InferenceResult:
                return InferenceResult(
                    effect=np.zeros(len(X)),
                    stderr=np.ones(len(X)),
                    method=inference_type,
                )

        est = EstimatorWithInference()
        assert isinstance(est, InferenceProvider)

    def test_estimator_without_inference_does_not_implement_protocol(self):
        """Test that estimator without effect_inference doesn't implement InferenceProvider."""
        from caml.estimators import InferenceProvider

        class EstimatorWithoutInference:
            pass

        est = EstimatorWithoutInference()
        assert not isinstance(est, InferenceProvider)
