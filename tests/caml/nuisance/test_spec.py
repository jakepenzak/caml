"""Tests for caml.nuisance.spec module."""

import pytest

from caml.nuisance import NuisanceTunerSpec

pytestmark = pytest.mark.nuisance


# ==============================================================================
# CREATION TESTS
# ==============================================================================


class TestNuisanceTunerSpecCreation:
    """Test NuisanceTunerSpec creation and initialization."""

    def test_create_with_defaults(self):
        """Test creation with all defaults."""
        spec = NuisanceTunerSpec()

        assert spec.fit_treatment_model is None
        assert spec.fit_outcome_model is None
        assert spec.fit_regression_model is None
        assert spec.treatment_model_config is None
        assert spec.outcome_model_config is None
        assert spec.regression_model_config is None

    def test_create_with_fit_flags(self):
        """Test creation with fit flags specified."""
        spec = NuisanceTunerSpec(
            fit_treatment_model=True,
            fit_outcome_model=True,
            fit_regression_model=False,
        )

        assert spec.fit_treatment_model is True
        assert spec.fit_outcome_model is True
        assert spec.fit_regression_model is False

    def test_create_with_configs(self):
        """Test creation with model configurations."""
        treatment_config = {"estimator_list": ["lgbm", "rf"], "time_budget": 60}
        outcome_config = {"estimator_list": ["xgboost"], "time_budget": 120}

        spec = NuisanceTunerSpec(
            treatment_model_config=treatment_config,
            outcome_model_config=outcome_config,
        )

        assert spec.treatment_model_config == treatment_config
        assert spec.outcome_model_config == outcome_config
        assert spec.regression_model_config is None

    def test_create_with_all_parameters(self):
        """Test creation with all parameters specified."""
        spec = NuisanceTunerSpec(
            fit_treatment_model=True,
            fit_outcome_model=False,
            fit_regression_model=True,
            treatment_model_config={"estimator_list": ["lgbm"]},
            outcome_model_config={"estimator_list": ["rf"]},
            regression_model_config={"estimator_list": ["xgboost"]},
        )

        assert spec.fit_treatment_model is True
        assert spec.fit_outcome_model is False
        assert spec.fit_regression_model is True
        assert spec.treatment_model_config["estimator_list"] == ["lgbm"]
        assert spec.outcome_model_config["estimator_list"] == ["rf"]
        assert spec.regression_model_config["estimator_list"] == ["xgboost"]


# ==============================================================================
# CONFIGURATION TESTS
# ==============================================================================


class TestNuisanceTunerSpecConfiguration:
    """Test configuration options."""

    def test_config_with_ray(self):
        """Test configuration with Ray support."""
        config = {"use_ray": True, "n_concurrent_trials": 4}
        spec = NuisanceTunerSpec(treatment_model_config=config)

        assert spec.treatment_model_config["use_ray"] is True
        assert spec.treatment_model_config["n_concurrent_trials"] == 4

    def test_config_with_spark(self):
        """Test configuration with Spark support."""
        config = {"use_spark": True, "n_concurrent_trials": 4}
        spec = NuisanceTunerSpec(outcome_model_config=config)

        assert spec.outcome_model_config["use_spark"] is True

    def test_config_with_estimator_list(self):
        """Test configuration with custom estimator list."""
        config = {"estimator_list": ["lgbm", "rf", "extra_tree"]}
        spec = NuisanceTunerSpec(regression_model_config=config)

        assert len(spec.regression_model_config["estimator_list"]) == 3
        assert "lgbm" in spec.regression_model_config["estimator_list"]

    def test_config_empty_dict(self):
        """Test with empty configuration dictionary."""
        spec = NuisanceTunerSpec(treatment_model_config={})

        assert spec.treatment_model_config == {}


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_flags_false(self):
        """Test with all fit flags set to False."""
        spec = NuisanceTunerSpec(
            fit_treatment_model=False,
            fit_outcome_model=False,
            fit_regression_model=False,
        )

        assert spec.fit_treatment_model is False
        assert spec.fit_outcome_model is False
        assert spec.fit_regression_model is False

    def test_all_flags_true(self):
        """Test with all fit flags set to True."""
        spec = NuisanceTunerSpec(
            fit_treatment_model=True,
            fit_outcome_model=True,
            fit_regression_model=True,
        )

        assert spec.fit_treatment_model is True
        assert spec.fit_outcome_model is True
        assert spec.fit_regression_model is True

    def test_mixed_flags_and_configs(self):
        """Test with mixed flags and configs."""
        spec = NuisanceTunerSpec(
            fit_treatment_model=True,
            fit_outcome_model=None,
            treatment_model_config={"time_budget": 60},
            outcome_model_config=None,
        )

        assert spec.fit_treatment_model is True
        assert spec.fit_outcome_model is None
        assert spec.treatment_model_config is not None
        assert spec.outcome_model_config is None
