"""Tests for caml.estimators.wrappers.base_wrapper module."""

import numpy as np
import pytest
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators import EstimatorCapabilities
from caml.estimators.wrappers.base_wrapper import BaseEconMLWrapperMixin
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.inference import InferenceResult, InferenceType

pytestmark = pytest.mark.estimators


@pytest.fixture
def binary_continuous_data():
    """Fixture for binary treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(n_obs=200, n_cont_confounders=5, seed=42)
    data = CausalDataset.from_dataframe(
        gen.df,
        X=[f"W{i}_continuous" for i in range(1, 6)],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )
    return data


# ==============================================================================
# CONCRETE WRAPPER IMPLEMENTATION TESTS
# ==============================================================================


class TestConcreteWrapperImplementation:
    """Test creating a concrete wrapper from BaseEconMLWrapperMixin."""

    def test_can_create_concrete_wrapper(self):
        """Test that a concrete wrapper can be created."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self, **econml_kwargs):
                self._econml_kwargs = econml_kwargs
                self._estimator = LinearDML(**econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                X = np.asarray(data.X)
                T = np.asarray(data.T).reshape(-1, 1)
                Y = np.asarray(data.Y)
                self._estimator.fit(Y, T, X=X)
                self._is_fitted = True
                return self

        # Should not raise
        wrapper = TestWrapper()
        assert hasattr(wrapper, "_estimator")
        assert hasattr(wrapper, "capabilities")


# ==============================================================================
# EFFECT METHOD TESTS
# ==============================================================================


class TestEffectMethod:
    """Test effect() method from BaseEconMLWrapperMixin."""

    def test_effect_delegates_to_underlying_estimator(self, binary_continuous_data):
        """Test that effect() delegates to underlying EconML estimator."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self):
                self._econml_kwargs = {
                    "model_y": RandomForestRegressor(n_estimators=10, random_state=42),
                    "model_t": RandomForestRegressor(n_estimators=10, random_state=42),
                }
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                X = np.asarray(data.X)
                T = np.asarray(data.T).reshape(-1, 1)
                Y = np.asarray(data.Y)
                self._estimator.fit(Y, T, X=X)
                self._is_fitted = True
                return self

        wrapper = TestWrapper()
        wrapper.fit(binary_continuous_data)
        result = wrapper.effect(binary_continuous_data.X[:10])

        assert isinstance(result, np.ndarray)
        assert len(result) == 10

    def test_effect_raises_if_not_fitted(self):
        """Test that effect() raises error if not fitted."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self):
                self._econml_kwargs = {}
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

        wrapper = TestWrapper()
        with pytest.raises(RuntimeError, match="must be fitted"):
            wrapper.effect(np.random.randn(10, 3))


# ==============================================================================
# EFFECT INFERENCE METHOD TESTS
# ==============================================================================


class TestEffectInferenceMethod:
    """Test effect_inference() method from BaseEconMLWrapperMixin."""

    def test_effect_inference_returns_inference_result(self, binary_continuous_data):
        """Test that effect_inference() returns InferenceResult."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self):
                self._econml_kwargs = {
                    "model_y": RandomForestRegressor(n_estimators=10, random_state=42),
                    "model_t": RandomForestRegressor(n_estimators=10, random_state=42),
                }
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                X = np.asarray(data.X)
                T = np.asarray(data.T).reshape(-1, 1)
                Y = np.asarray(data.Y)
                self._estimator.fit(Y, T, X=X)
                self._is_fitted = True
                return self

        wrapper = TestWrapper()
        wrapper.fit(binary_continuous_data)
        result = wrapper.effect_inference(binary_continuous_data.X[:10])

        assert isinstance(result, InferenceResult)
        assert hasattr(result, "effect")
        assert hasattr(result, "stderr")
        assert len(result.effect) == 10

    def test_effect_inference_bootstrap_not_implemented(self, binary_continuous_data):
        """Test that bootstrap inference raises NotImplementedError."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self):
                self._econml_kwargs = {}
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

        wrapper = TestWrapper()
        wrapper.fit(binary_continuous_data)

        with pytest.raises(NotImplementedError, match="Bootstrap inference"):
            wrapper.effect_inference(
                binary_continuous_data.X[:10], inference_type=InferenceType.BOOTSTRAP
            )


# ==============================================================================
# GET/SET PARAMS TESTS
# ==============================================================================


class TestGetSetParams:
    """Test get_params() and set_params() methods."""

    def test_get_params_returns_econml_kwargs(self):
        """Test that get_params() returns EconML kwargs."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self, discrete_treatment=True, **econml_kwargs):
                self._econml_kwargs = {"discrete_treatment": discrete_treatment}
                self._econml_kwargs.update(econml_kwargs)
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

        wrapper = TestWrapper(discrete_treatment=True)
        params = wrapper.get_params()

        assert isinstance(params, dict)
        assert "discrete_treatment" in params
        assert params["discrete_treatment"] is True

    def test_set_params_updates_parameters(self):
        """Test that set_params() updates parameters."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self, discrete_treatment=True):
                self._econml_kwargs = {"discrete_treatment": discrete_treatment}
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

        wrapper = TestWrapper(discrete_treatment=True)
        wrapper.set_params(discrete_treatment=False)

        assert wrapper._econml_kwargs["discrete_treatment"] is False

    def test_set_params_resets_fitted_state(self):
        """Test that set_params() resets fitted state."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self, discrete_treatment=True):
                self._econml_kwargs = {"discrete_treatment": discrete_treatment}
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

        wrapper = TestWrapper()
        wrapper._is_fitted = True
        wrapper.set_params(discrete_treatment=False)

        assert wrapper._is_fitted is False


# ==============================================================================
# ATTRIBUTE FORWARDING TESTS
# ==============================================================================


class TestAttributeForwarding:
    """Test __getattr__ forwarding to underlying estimator."""

    def test_forwards_attribute_to_underlying_estimator(self):
        """Test that attributes are forwarded to underlying estimator."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self):
                self._econml_kwargs = {"discrete_treatment": True}
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

        wrapper = TestWrapper()
        # Access an attribute that exists on LinearDML but not on TestWrapper
        assert hasattr(wrapper, "discrete_treatment")
        assert wrapper.discrete_treatment is True

    def test_raises_attribute_error_if_not_found(self):
        """Test that AttributeError is raised for non-existent attributes."""
        from caml.automl import SearchSpace

        class TestWrapper(BaseEconMLWrapperMixin):
            capabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types={InferenceType.ANALYTIC},
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                supports_inference=True,
            )

            default_search_space: SearchSpace = ()

            def __init__(self):
                self._econml_kwargs = {}
                self._estimator = LinearDML(**self._econml_kwargs)
                self._is_fitted = False

            def fit(self, data, **kwargs):
                self._is_fitted = True
                return self

        wrapper = TestWrapper()
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = wrapper.nonexistent_attribute
