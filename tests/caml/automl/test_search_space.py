"""Tests for caml.automl.search_space module."""

import pytest

from caml.automl.search_space import (
    BoolSpec,
    CategoricalSpec,
    ConstantSpec,
    FloatSpec,
    IntSpec,
    NuisanceModelSpec,
    SearchSpace,
    StandardMLSpec,
)

pytestmark = [pytest.mark.automl]


# ==============================================================================
# INT SPEC TESTS
# ==============================================================================


class TestIntSpec:
    """Test IntSpec functionality."""

    def test_valid_linear_scale(self):
        """Test valid integer spec with linear scale."""
        spec = IntSpec(name="cv", lower=2, upper=10)
        assert spec.name == "cv"
        assert spec.lower == 2
        assert spec.upper == 10
        assert not spec.log

    def test_valid_log_scale(self):
        """Test valid integer spec with log scale."""
        spec = IntSpec(name="max_iter", lower=100, upper=10000, log=True)
        assert spec.log
        assert spec.lower == 100
        assert spec.upper == 10000

    def test_with_step(self):
        """Test integer spec with step parameter."""
        spec = IntSpec(name="n_estimators", lower=10, upper=100, step=10)
        assert spec.step == 10

    def test_invalid_range_raises_error(self):
        """Test that lower >= upper raises ValueError."""
        with pytest.raises(ValueError, match="must be < upper"):
            IntSpec(name="bad", lower=10, upper=5)

    def test_equal_bounds_raises_error(self):
        """Test that lower == upper raises ValueError."""
        with pytest.raises(ValueError, match="must be < upper"):
            IntSpec(name="bad", lower=5, upper=5)

    def test_log_scale_with_zero_lower_raises_error(self):
        """Test that log scale with non-positive lower bound raises error."""
        with pytest.raises(ValueError, match="Log scale requires positive lower bound"):
            IntSpec(name="bad", lower=0, upper=100, log=True)

    def test_log_scale_with_negative_lower_raises_error(self):
        """Test that log scale with negative lower bound raises error."""
        with pytest.raises(ValueError, match="Log scale requires positive lower bound"):
            IntSpec(name="bad", lower=-10, upper=100, log=True)

    def test_to_optuna_linear(self):
        """Test Optuna conversion for linear scale."""
        spec = IntSpec(name="cv", lower=2, upper=10)

        class MockTrial:
            def suggest_int(self, name, lower, upper, log=False, step=None):
                assert name == "cv"
                assert lower == 2
                assert upper == 10
                assert not log
                assert step is None
                return 5

        result = spec.to_optuna(MockTrial())
        assert result == 5

    def test_to_optuna_log_with_step(self):
        """Test Optuna conversion with log scale and step."""
        spec = IntSpec(name="max_iter", lower=100, upper=1000, log=True, step=50)

        class MockTrial:
            def suggest_int(self, name, lower, upper, log=False, step=None):
                assert name == "max_iter"
                assert lower == 100
                assert upper == 1000
                assert log
                assert step == 50
                return 500

        result = spec.to_optuna(MockTrial())
        assert result == 500


# ==============================================================================
# FLOAT SPEC TESTS
# ==============================================================================


class TestFloatSpec:
    """Test FloatSpec functionality."""

    def test_valid_linear_scale(self):
        """Test valid float spec with linear scale."""
        spec = FloatSpec(name="alpha", lower=0.0, upper=1.0)
        assert spec.name == "alpha"
        assert spec.lower == 0.0
        assert spec.upper == 1.0
        assert not spec.log

    def test_valid_log_scale(self):
        """Test valid float spec with log scale."""
        spec = FloatSpec(name="lambda", lower=1e-5, upper=1e-1, log=True)
        assert spec.log
        assert spec.lower == 1e-5
        assert spec.upper == 1e-1

    def test_with_step(self):
        """Test float spec with step parameter."""
        spec = FloatSpec(name="learning_rate", lower=0.0, upper=1.0, step=0.1)
        assert spec.step == 0.1

    def test_invalid_range_raises_error(self):
        """Test that lower >= upper raises ValueError."""
        with pytest.raises(ValueError, match="must be < upper"):
            FloatSpec(name="bad", lower=1.0, upper=0.5)

    def test_log_scale_with_zero_lower_raises_error(self):
        """Test that log scale with zero lower bound raises error."""
        with pytest.raises(ValueError, match="Log scale requires positive lower bound"):
            FloatSpec(name="bad", lower=0.0, upper=1.0, log=True)

    def test_to_optuna_linear(self):
        """Test Optuna conversion for linear scale."""
        spec = FloatSpec(name="alpha", lower=0.0, upper=1.0)

        class MockTrial:
            def suggest_float(self, name, lower, upper, log=False, step=None):
                assert name == "alpha"
                assert lower == 0.0
                assert upper == 1.0
                assert not log
                assert step is None
                return 0.5

        result = spec.to_optuna(MockTrial())
        assert result == 0.5

    def test_to_optuna_log_with_step(self):
        """Test Optuna conversion with log scale and step."""
        spec = FloatSpec(name="reg", lower=1e-4, upper=1e-1, log=True, step=1e-5)

        class MockTrial:
            def suggest_float(self, name, lower, upper, log=False, step=None):
                assert name == "reg"
                assert lower == 1e-4
                assert upper == 1e-1
                assert log
                assert step == 1e-5
                return 1e-3

        result = spec.to_optuna(MockTrial())
        assert result == 1e-3


# ==============================================================================
# CATEGORICAL SPEC TESTS
# ==============================================================================


class TestCategoricalSpec:
    """Test CategoricalSpec functionality."""

    def test_valid_string_choices(self):
        """Test categorical spec with string choices."""
        spec = CategoricalSpec(name="solver", choices=["auto", "svd", "cholesky"])
        assert spec.name == "solver"
        assert spec.choices == ["auto", "svd", "cholesky"]

    def test_valid_mixed_choices(self):
        """Test categorical spec with mixed type choices."""
        spec = CategoricalSpec(name="param", choices=[1, 2.5, "value", None])
        assert len(spec.choices) == 4

    def test_empty_choices_raises_error(self):
        """Test that empty choices list raises ValueError."""
        with pytest.raises(ValueError, match="Empty choices list"):
            CategoricalSpec(name="bad", choices=[])

    def test_duplicate_choices_raises_error(self):
        """Test that duplicate choices raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate choices"):
            CategoricalSpec(name="bad", choices=["a", "b", "a"])

    def test_duplicate_numeric_choices_raises_error(self):
        """Test that duplicate numeric choices raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate choices"):
            CategoricalSpec(name="bad", choices=[1, 2, 1])

    def test_to_optuna(self):
        """Test Optuna conversion."""
        spec = CategoricalSpec(name="solver", choices=["auto", "svd", "cholesky"])

        class MockTrial:
            def suggest_categorical(self, name, choices):
                assert name == "solver"
                assert choices == ["auto", "svd", "cholesky"]
                return "svd"

        result = spec.to_optuna(MockTrial())
        assert result == "svd"


# ==============================================================================
# BOOL SPEC TESTS
# ==============================================================================


class TestBoolSpec:
    """Test BoolSpec functionality."""

    def test_valid_bool_spec(self):
        """Test valid boolean spec."""
        spec = BoolSpec(name="fit_intercept")
        assert spec.name == "fit_intercept"

    def test_validate_does_not_raise(self):
        """Test that validate method succeeds."""
        spec = BoolSpec(name="normalize")
        spec.validate()  # Should not raise

    def test_to_optuna(self):
        """Test Optuna conversion."""
        spec = BoolSpec(name="fit_intercept")

        class MockTrial:
            def suggest_categorical(self, name, choices):
                assert name == "fit_intercept"
                assert choices == [True, False]
                return True

        result = spec.to_optuna(MockTrial())
        assert result is True


# ==============================================================================
# CONSTANT SPEC TESTS
# ==============================================================================


class TestConstantSpec:
    """Test ConstantSpec functionality."""

    def test_valid_int_constant(self):
        """Test constant spec with integer value."""
        spec = ConstantSpec(name="random_state", value=42)
        assert spec.name == "random_state"
        assert spec.value == 42

    def test_valid_string_constant(self):
        """Test constant spec with string value."""
        spec = ConstantSpec(name="method", value="default")
        assert spec.value == "default"

    def test_valid_none_constant(self):
        """Test constant spec with None value."""
        spec = ConstantSpec(name="optional_param", value=None)
        assert spec.value is None

    def test_validate_does_not_raise(self):
        """Test that validate method succeeds."""
        spec = ConstantSpec(name="param", value="value")
        spec.validate()  # Should not raise

    def test_to_optuna_returns_constant(self):
        """Test that to_optuna returns the constant value."""
        spec = ConstantSpec(name="random_state", value=42)

        class MockTrial:
            pass

        result = spec.to_optuna(MockTrial())
        assert result == 42


# ==============================================================================
# NUISANCE MODEL SPEC TESTS
# ==============================================================================


class TestNuisanceModelSpec:
    """Test NuisanceModelSpec functionality."""

    def test_valid_outcome_model(self):
        """Test nuisance model spec for outcome model."""
        spec = NuisanceModelSpec(name="model_y", model_type="outcome")
        assert spec.name == "model_y"
        assert spec.model_type == "outcome"

    def test_valid_treatment_model(self):
        """Test nuisance model spec for treatment model."""
        spec = NuisanceModelSpec(name="model_t", model_type="treatment")
        assert spec.model_type == "treatment"

    def test_valid_regression_model(self):
        """Test nuisance model spec for regression model."""
        spec = NuisanceModelSpec(name="model_regression", model_type="regression")
        assert spec.model_type == "regression"

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            NuisanceModelSpec(name="bad", model_type="invalid")  # type: ignore[arg-type]

    def test_to_optuna_raises_not_implemented(self):
        """Test that to_optuna raises NotImplementedError."""
        spec = NuisanceModelSpec(name="model_y", model_type="outcome")

        class MockTrial:
            pass

        with pytest.raises(NotImplementedError, match="handled separately"):
            spec.to_optuna(MockTrial())


# ==============================================================================
# SEARCH SPACE TYPE TESTS
# ==============================================================================


class TestSearchSpace:
    """Test SearchSpace type alias and integration."""

    def test_search_space_is_sequence_of_specs(self):
        """Test that SearchSpace type works as expected."""
        search_space: SearchSpace = (
            IntSpec(name="cv", lower=2, upper=10),
            FloatSpec(name="alpha", lower=0.0, upper=1.0),
            CategoricalSpec(name="solver", choices=["auto", "svd"]),
            BoolSpec(name="fit_intercept"),
            ConstantSpec(name="random_state", value=42),
            NuisanceModelSpec(name="model_y", model_type="outcome"),
            StandardMLSpec(name="model_final", models=["lightgbm"]),
        )

        assert len(search_space) == 7
        assert isinstance(search_space[0], IntSpec)
        assert isinstance(search_space[1], FloatSpec)
        assert isinstance(search_space[2], CategoricalSpec)
        assert isinstance(search_space[3], BoolSpec)
        assert isinstance(search_space[4], ConstantSpec)
        assert isinstance(search_space[5], NuisanceModelSpec)
        assert isinstance(search_space[6], StandardMLSpec)

    def test_all_specs_have_name_attribute(self):
        """Test that all specs have name attribute."""
        search_space: SearchSpace = (
            IntSpec(name="int_param", lower=1, upper=10),
            FloatSpec(name="float_param", lower=0.0, upper=1.0),
            CategoricalSpec(name="cat_param", choices=["a", "b"]),
            BoolSpec(name="bool_param"),
            ConstantSpec(name="const_param", value=42),
            NuisanceModelSpec(name="model_param", model_type="outcome"),
            StandardMLSpec(name="ml_param", models=["lightgbm"]),
        )

        expected_names = [
            "int_param",
            "float_param",
            "cat_param",
            "bool_param",
            "const_param",
            "model_param",
            "ml_param",
        ]
        actual_names = [spec.name for spec in search_space]
        assert actual_names == expected_names

    def test_all_specs_have_validate_method(self):
        """Test that all specs have validate method."""
        search_space: SearchSpace = (
            IntSpec(name="cv", lower=2, upper=10),
            FloatSpec(name="alpha", lower=0.0, upper=1.0),
            CategoricalSpec(name="solver", choices=["auto", "svd"]),
            BoolSpec(name="fit_intercept"),
            ConstantSpec(name="random_state", value=42),
        )

        for spec in search_space:
            spec.validate()  # Should not raise


# ==============================================================================
# STANDARD ML SPEC TESTS
# ==============================================================================


class TestStandardMLSpec:
    """Test StandardMLSpec functionality."""

    def test_valid_with_explicit_models(self):
        """Test standard ML spec with explicitly provided models."""
        spec = StandardMLSpec(name="model_final", models=["lightgbm", "xgboost"])
        assert spec.name == "model_final"
        assert spec.models == ["lightgbm", "xgboost"]

    def test_valid_with_all_models(self):
        """Test standard ML spec defaults to all available models."""
        spec = StandardMLSpec(name="model_final")
        assert spec.models is not None
        assert len(spec.models) > 0
        # Should include all registered standard ML models
        assert "lightgbm" in spec.models
        assert "xgboost" in spec.models
        assert "random_forest" in spec.models

    def test_valid_with_single_model(self):
        """Test standard ML spec with single model."""
        spec = StandardMLSpec(name="model_final", models=["lightgbm"])
        assert spec.models == ["lightgbm"]

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_types"):
            StandardMLSpec(name="bad", models=["invalid_model"])

    def test_invalid_mixed_models_raises_error(self):
        """Test that mixed valid/invalid models raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model_types"):
            StandardMLSpec(name="bad", models=["lightgbm", "invalid_model"])

    def test_to_optuna_raises_not_implemented(self):
        """Test that to_optuna raises NotImplementedError."""
        spec = StandardMLSpec(name="model_final", models=["lightgbm"])

        class MockTrial:
            pass

        with pytest.raises(NotImplementedError, match="handled separately"):
            spec.to_optuna(MockTrial())

    def test_valid_keys_populated_correctly(self):
        """Test that _VALID_KEYS is populated from registry."""
        spec = StandardMLSpec(name="model_final", models=["lightgbm"])
        assert hasattr(spec, "_VALID_KEYS")
        assert len(spec._VALID_KEYS) > 0
        assert "lightgbm" in spec._VALID_KEYS
