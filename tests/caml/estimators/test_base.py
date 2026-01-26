"""Tests for caml.estimators.base."""

import numpy as np
import pandas as pd
import pytest

from caml.data.data_schema import Estimand, OutcomeType, TreatmentType
from caml.data.dataset import CausalDataset
from caml.estimators.base import (
    AutoCateEstimator,
    BaseWrapperMixin,
    EstimatorCapabilities,
    InferenceProvider,
)
from caml.inference.inference_schema import InferenceType
from caml.inference.results import InferenceResult


class TestEstimatorCapabilities:
    """Tests for EstimatorCapabilities dataclass."""

    def test_creation_minimal(self):
        """Test creating EstimatorCapabilities with minimal args."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE, Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        assert TreatmentType.BINARY in caps.treatment_types
        assert OutcomeType.CONTINUOUS in caps.outcome_types
        assert InferenceType.BOOTSTRAP in caps.inference_types
        assert Estimand.ATE in caps.estimands
        assert caps.supports_weights is False
        assert caps.supports_inference is False

    def test_creation_full(self):
        """Test creating EstimatorCapabilities with all args."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP, InferenceType.ANALYTIC},
            estimands={Estimand.ATE, Estimand.CATE},
            supports_controls_in_first_stage_only=True,
            supports_weights=True,
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            supports_inference=True,
        )

        assert len(caps.treatment_types) == 2
        assert caps.supports_weights is True
        assert caps.supports_inference is True
        assert caps.requires_treatment_model is True
        assert caps.requires_outcome_model is True
        assert caps.supports_controls_in_first_stage_only is True

    def test_frozen_dataclass(self):
        """Test EstimatorCapabilities is frozen (immutable)."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            caps.supports_weights = True

    def test_is_compatible_binary_continuous(self):
        """Test is_compatible with binary treatment, continuous outcome."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert caps.is_compatible(data) is True

    def test_is_compatible_incompatible_treatment(self):
        """Test is_compatible returns False for incompatible treatment type."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        X = np.array([[1, 2], [3, 4]])
        T = np.array([0.1, 0.5])
        Y = np.array([1.0, 2.0])
        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert caps.is_compatible(data) is False

    def test_is_compatible_incompatible_outcome(self):
        """Test is_compatible returns False for incompatible outcome type."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([0, 1])
        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.BINARY,
        )

        assert caps.is_compatible(data) is False

    def test_is_compatible_multiple_types(self):
        """Test is_compatible with multiple supported types."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False,
        )

        # Binary treatment, continuous outcome
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        data1 = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )
        assert caps.is_compatible(data1) is True

        # Continuous treatment, binary outcome
        T2 = np.array([0.1, 0.5])
        Y2 = np.array([0, 1])
        data2 = CausalDataset(
            X=X,
            T=T2,
            Y=Y2,
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.BINARY,
        )
        assert caps.is_compatible(data2) is True


class TestAutoCateEstimatorProtocol:
    """Tests for AutoCateEstimator Protocol."""

    def test_protocol_implementation_minimal(self):
        """Test implementing minimal AutoCateEstimator protocol."""

        class MinimalEstimator:
            @property
            def capabilities(self):
                return EstimatorCapabilities(
                    treatment_types={TreatmentType.BINARY},
                    outcome_types={OutcomeType.CONTINUOUS},
                    inference_types={InferenceType.BOOTSTRAP},
                    estimands={Estimand.CATE},
                    supports_controls_in_first_stage_only=False,
                    supports_weights=False,
                    requires_treatment_model=False,
                    requires_outcome_model=False,
                    requires_regression_model=False,
                    supports_inference=False,
                )

            @property
            def clean_name(self):
                return "Minimal Estimator"

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                temp = cls()
                return temp.capabilities.is_compatible(data)

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return self.capabilities.is_compatible(data)

            def fit(self, data: CausalDataset, **kwargs):
                return self

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                return np.zeros(len(X))

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params) -> dict:
                return self

        estimator = MinimalEstimator()
        assert isinstance(estimator, AutoCateEstimator)

    def test_protocol_implementation_missing_capabilities(self):
        """Test implementing AutoCateEstimator without capabilities fails."""

        class NoCapabilitiesEstimator:
            @property
            def clean_name(self):
                return "No Capabilities"

            def fit(self, data: CausalDataset, **kwargs):
                return self

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                return np.zeros(len(X))

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params) -> dict:
                return self

        estimator = NoCapabilitiesEstimator()
        assert not isinstance(estimator, AutoCateEstimator)


class TestInferenceProviderProtocol:
    """Tests for InferenceProvider Protocol."""

    def test_protocol_implementation(self):
        """Test implementing InferenceProvider protocol."""

        class InferenceCapableEstimator:
            def effect_inference(
                self,
                X: np.ndarray | pd.DataFrame,
                inference_type: InferenceType | None = None,
                bootstrapper: bool | None = None,
                **effect_inference_kwargs,
            ) -> InferenceResult:
                n = len(X) if isinstance(X, (np.ndarray, pd.DataFrame)) else X.shape[0]
                return InferenceResult(
                    effect=np.ones(n),
                    stderr=np.ones(n) * 0.1,
                    method=InferenceType.ANALYTIC,
                )

        estimator = InferenceCapableEstimator()
        assert isinstance(estimator, InferenceProvider)

    def test_protocol_missing_effect_inference(self):
        """Test class without effect_inference doesn't satisfy protocol."""

        class NoInferenceEstimator:
            def some_other_method(self):
                pass

        estimator = NoInferenceEstimator()
        assert not isinstance(estimator, InferenceProvider)


class TestBaseWrapperMixin:
    """Tests for BaseWrapperMixin."""

    def test_is_compatible_with_classmethod(self):
        """Test is_compatible_with class method works without instantiation."""

        class ConcreteWrapper(BaseWrapperMixin):
            @property
            def clean_name(self) -> str:
                return "ConcreteWrapper"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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

            def fit(self, data: CausalDataset, **fit_kwargs):
                return self

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params):
                return self

        # Create compatible data
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        # Test without instantiation
        assert ConcreteWrapper.is_compatible_with(data) is True

    def test_check_compatibility_raises_on_incompatible(self):
        """Test check_compatibility raises detailed ValueError when incompatible."""

        class ConcreteWrapper(BaseWrapperMixin):
            @property
            def clean_name(self) -> str:
                return "ConcreteWrapper"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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

            def fit(self, data: CausalDataset, **fit_kwargs):
                return self

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params):
                return self

        # Create incompatible data (binary outcome instead of continuous)
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([0, 1])
        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.BINARY,
        )

        estimator = ConcreteWrapper()

        # Should raise with detailed message
        with pytest.raises(ValueError, match="Data incompatible"):
            estimator.check_compatibility(data, raise_error=True)

    def test_check_compatibility_returns_bool_when_raise_error_false(self):
        """Test check_compatibility returns boolean when raise_error=False."""

        class ConcreteWrapper(BaseWrapperMixin):
            @property
            def clean_name(self) -> str:
                return "ConcreteWrapper"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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

            def fit(self, data: CausalDataset, **fit_kwargs):
                return self

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params):
                return self

        # Create incompatible data
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([0, 1])
        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.BINARY,
        )

        estimator = ConcreteWrapper()

        # Should return False, not raise
        result = estimator.check_compatibility(data, raise_error=False)
        assert result is False

    def test_effect_raises_before_fit(self):
        """Test effect raises RuntimeError if called before fit."""

        class ConcreteWrapper(BaseWrapperMixin):
            def __init__(self):
                self._estimator = None
                self._is_fitted = False

            @property
            def clean_name(self) -> str:
                return "ConcreteWrapper"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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

            def fit(self, data: CausalDataset, **fit_kwargs):
                self._is_fitted = True
                return self

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params):
                return self

        estimator = ConcreteWrapper()
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(RuntimeError, match="must be fitted"):
            estimator.effect(X)

    def test_getattr_delegation(self):
        """Test __getattr__ properly delegates to underlying estimator."""

        class MockEconMLEstimator:
            def some_econml_method(self):
                return "econml_result"

            econml_attribute = "econml_value"

        class ConcreteWrapper(BaseWrapperMixin):
            def __init__(self):
                self._estimator = MockEconMLEstimator()
                self._is_fitted = False

            @property
            def clean_name(self) -> str:
                return "ConcreteWrapper"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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

            def fit(self, data: CausalDataset, **fit_kwargs):
                return self

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params):
                return self

        estimator = ConcreteWrapper()

        # Should delegate to underlying estimator
        assert estimator.some_econml_method() == "econml_result"
        assert estimator.econml_attribute == "econml_value"

    def test_getattr_raises_for_nonexistent(self):
        """Test __getattr__ raises AttributeError for non-existent attributes."""

        class ConcreteWrapper(BaseWrapperMixin):
            def __init__(self):
                self._estimator = None

            @property
            def clean_name(self) -> str:
                return "ConcreteWrapper"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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

            def fit(self, data: CausalDataset, **fit_kwargs):
                return self

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params):
                return self

        estimator = ConcreteWrapper()

        with pytest.raises(AttributeError, match="has no attribute"):
            estimator.nonexistent_attribute
