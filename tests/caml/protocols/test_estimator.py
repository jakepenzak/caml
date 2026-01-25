"""Tests for caml.protocols.estimator."""

import numpy as np
import pandas as pd
import pytest

from caml.data.data_schema import Estimand, OutcomeType, TreatmentType
from caml.data.dataset import CausalDataset
from caml.inference.inference_schema import InferenceType
from caml.protocols.estimator import AutoCateEstimator, EstimatorCapabilities


class TestEstimatorCapabilities:
    """Tests for EstimatorCapabilities dataclass."""

    def test_creation_minimal(self):
        """Test creating EstimatorCapabilities with minimal args."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE, Estimand.CATE},
        )

        assert TreatmentType.BINARY in caps.treatment_types
        assert OutcomeType.CONTINUOUS in caps.outcome_types
        assert InferenceType.BOOTSTRAP in caps.inference_types
        assert Estimand.ATE in caps.estimands
        assert caps.requires_propensity is False
        assert caps.supports_inference is False

    def test_creation_full(self):
        """Test creating EstimatorCapabilities with all args."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP, InferenceType.ANALYTIC},
            estimands={Estimand.ATE, Estimand.CATE},
            requires_propensity=True,
            supports_inference=True,
        )

        assert len(caps.treatment_types) == 2
        assert caps.requires_propensity is True
        assert caps.supports_inference is True

    def test_frozen_dataclass(self):
        """Test EstimatorCapabilities is frozen (immutable)."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE},
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            caps.requires_propensity = True

    def test_is_compatible_binary_continuous(self):
        """Test is_compatible with binary treatment, continuous outcome."""
        caps = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types={InferenceType.BOOTSTRAP},
            estimands={Estimand.ATE},
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
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.BOOTSTRAP},
                estimands={Estimand.CATE},
            )
            clean_name = "Minimal Estimator"

            def __init__(self):
                pass

            def fit(self, data: CausalDataset, **kwargs):
                return self

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                return np.zeros(len(X))

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params) -> dict:
                return {}

        estimator = MinimalEstimator()
        assert isinstance(estimator, AutoCateEstimator)

    def test_protocol_implementation_missing_capabilities(self):
        """Test implementing AutoCateEstimator without capabilities fails."""

        class NoCapabilitiesEstimator:
            def fit(self, data: CausalDataset, **kwargs):
                return self

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                return np.zeros(len(X))

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params) -> dict:
                return {}

        estimator = NoCapabilitiesEstimator()
        assert not isinstance(estimator, AutoCateEstimator)

    def test_protocol_implementation_missing_fit(self):
        """Test implementing AutoCateEstimator without fit fails."""

        class NoFitEstimator:
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.BOOTSTRAP},
                estimands={Estimand.CATE},
            )
            clean_name = "NoFitEstimator"

            def __init__(self):
                pass

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                return np.zeros(len(X))

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params) -> dict:
                return {}

        estimator = NoFitEstimator()
        assert not isinstance(estimator, AutoCateEstimator)

    def test_protocol_implementation_missing_effect(self):
        """Test implementing AutoCateEstimator without effect fails."""

        class NoEffectEstimator:
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.BOOTSTRAP},
                estimands={Estimand.CATE},
            )
            clean_name = "No Effect Estimator"

            def __init__(self):
                pass

            def fit(self, data: CausalDataset, **kwargs):
                return self

            def get_params(self, deep: bool = True) -> dict:
                return {}

            def set_params(self, **params) -> dict:
                return {}

        estimator = NoEffectEstimator()
        assert not isinstance(estimator, AutoCateEstimator)

    def test_protocol_implementation_missing_get_params(self):
        """Test implementing AutoCateEstimator without get_params fails."""

        class NoGetParamsEstimator:
            def __init__(self):
                self.capabilities = EstimatorCapabilities(
                    treatment_types={TreatmentType.BINARY},
                    outcome_types={OutcomeType.CONTINUOUS},
                    inference_types={InferenceType.BOOTSTRAP},
                    estimands={Estimand.CATE},
                )
                self.clean_name = "No Get Params Estimator"

            def fit(self, data: CausalDataset, **kwargs):
                return self

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                return np.zeros(len(X))

            def set_params(self, **params) -> dict:
                return {}

        estimator = NoGetParamsEstimator()
        assert not isinstance(estimator, AutoCateEstimator)

    def test_protocol_implementation_missing_set_params(self):
        """Test implementing AutoCateEstimator without set_params fails."""

        class NoSetParamsEstimator:
            def __init__(self):
                self.capabilities = EstimatorCapabilities(
                    treatment_types={TreatmentType.BINARY},
                    outcome_types={OutcomeType.CONTINUOUS},
                    inference_types={InferenceType.BOOTSTRAP},
                    estimands={Estimand.CATE},
                )
                self.clean_name = "No Set Params Estimator"

            def fit(self, data: CausalDataset, **kwargs):
                return self

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                return np.zeros(len(X))

            def get_params(self, deep: bool = True) -> dict:
                return {}

        estimator = NoSetParamsEstimator()
        assert not isinstance(estimator, AutoCateEstimator)

    def test_protocol_full_implementation(self):
        """Test full AutoCateEstimator implementation with realistic behavior."""

        class FullEstimator:
            def __init__(self, param1=1.0, param2="default"):
                self.capabilities = EstimatorCapabilities(
                    treatment_types={TreatmentType.BINARY},
                    outcome_types={OutcomeType.CONTINUOUS},
                    inference_types={InferenceType.BOOTSTRAP},
                    estimands={Estimand.CATE},
                    requires_propensity=False,
                    supports_inference=True,
                )
                self.clean_name = "Full Estimator"
                self.param1 = param1
                self.param2 = param2
                self._is_fitted = False

            def fit(self, data: CausalDataset, **kwargs):
                self._is_fitted = True
                return self

            def effect(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray:
                if not self._is_fitted:
                    raise ValueError("Estimator must be fitted first")
                n = len(X) if isinstance(X, (np.ndarray, pd.DataFrame)) else X.shape[0]
                return np.ones(n) * self.param1

            def get_params(self, deep: bool = True) -> dict:
                return {"param1": self.param1, "param2": self.param2}

            def set_params(self, **params) -> dict:
                for key, value in params.items():
                    setattr(self, key, value)
                return params

        estimator = FullEstimator()
        assert isinstance(estimator, AutoCateEstimator)

        # Test fit
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        data = CausalDataset(X=X, T=T, Y=Y)
        estimator.fit(data)

        # Test effect
        effects = estimator.effect(X)
        assert effects.shape == (2,)
        np.testing.assert_array_equal(effects, np.array([1.0, 1.0]))

        # Test get_params
        params = estimator.get_params()
        assert params == {"param1": 1.0, "param2": "default"}

        # Test set_params
        estimator.set_params(param1=2.0)
        assert estimator.param1 == 2.0
