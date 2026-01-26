"""Tests for caml.nuisance.tuner module."""

import pytest

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.nuisance import NuisanceTuner, NuisanceTunerSpec

pytestmark = pytest.mark.nuisance


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def binary_treatment_continuous_outcome_data():
    """Generate dataset with binary treatment and continuous outcome."""
    gen = SyntheticDataGenerator(
        n_obs=100, n_cont_modifiers=3, n_binary_treatments=1, n_cont_outcomes=1, seed=42
    )

    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def continuous_treatment_continuous_outcome_data():
    """Generate dataset with continuous treatment and continuous outcome."""
    gen = SyntheticDataGenerator(
        n_obs=100, n_cont_modifiers=3, n_cont_treatments=1, n_cont_outcomes=1, seed=42
    )

    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_continuous",
        Y="Y1_continuous",
        treatment_type=TreatmentType.CONTINUOUS,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def binary_treatment_binary_outcome_data():
    """Generate dataset with binary treatment and binary outcome."""
    gen = SyntheticDataGenerator(
        n_obs=100,
        n_cont_modifiers=3,
        n_binary_treatments=1,
        n_binary_outcomes=1,
        seed=42,
    )

    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y2_binary",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.BINARY,
    )


# ==============================================================================
# INITIALIZATION TESTS
# ==============================================================================


class TestNuisanceTunerInitialization:
    """Test NuisanceTuner initialization."""

    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        tuner = NuisanceTuner()

        assert tuner.time_budget == 300
        assert tuner.use_ray is False
        assert tuner.use_spark is False
        assert tuner.seed is None
        assert tuner.verbose == 0
        assert tuner.treatment_model_ is None
        assert tuner.outcome_model_ is None
        assert tuner.regression_model_ is None

    def test_create_with_custom_parameters(self):
        """Test creation with custom parameters."""
        tuner = NuisanceTuner(time_budget=60, use_ray=True, seed=42, verbose=1)

        assert tuner.time_budget == 60
        assert tuner.use_ray is True
        assert tuner.seed == 42
        assert tuner.verbose == 1

    def test_create_with_spark(self):
        """Test creation with Spark enabled."""
        tuner = NuisanceTuner(use_spark=True)

        assert tuner.use_spark is True
        assert tuner.use_ray is False


# ==============================================================================
# FIT TESTS
# ==============================================================================


class TestNuisanceTunerFit:
    """Test NuisanceTuner fit functionality."""

    def test_fit_treatment_model_binary(self, binary_treatment_continuous_outcome_data):
        """Test fitting treatment model for binary treatment."""
        spec = NuisanceTunerSpec(fit_treatment_model=True)
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_continuous_outcome_data, spec)

        assert tuner.treatment_model_ is not None
        assert tuner.outcome_model_ is None
        assert tuner.regression_model_ is None

    def test_fit_outcome_model(self, binary_treatment_continuous_outcome_data):
        """Test fitting outcome model."""
        spec = NuisanceTunerSpec(fit_outcome_model=True)
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_continuous_outcome_data, spec)

        assert tuner.treatment_model_ is None
        assert tuner.outcome_model_ is not None
        assert tuner.regression_model_ is None

    def test_fit_regression_model(self, binary_treatment_continuous_outcome_data):
        """Test fitting regression model."""
        spec = NuisanceTunerSpec(fit_regression_model=True)
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_continuous_outcome_data, spec)

        assert tuner.treatment_model_ is None
        assert tuner.outcome_model_ is None
        assert tuner.regression_model_ is not None

    def test_fit_all_models(self, binary_treatment_continuous_outcome_data):
        """Test fitting all nuisance models."""
        spec = NuisanceTunerSpec(
            fit_treatment_model=True,
            fit_outcome_model=True,
            fit_regression_model=True,
        )
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_continuous_outcome_data, spec)

        assert tuner.treatment_model_ is not None
        assert tuner.outcome_model_ is not None
        assert tuner.regression_model_ is not None

    def test_fit_no_models(self, binary_treatment_continuous_outcome_data):
        """Test fitting with no models specified."""
        spec = NuisanceTunerSpec()
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_continuous_outcome_data, spec)

        assert tuner.treatment_model_ is None
        assert tuner.outcome_model_ is None
        assert tuner.regression_model_ is None


# ==============================================================================
# CONFIGURATION TESTS
# ==============================================================================


class TestNuisanceTunerConfiguration:
    """Test NuisanceTuner with custom configurations."""

    def test_fit_with_custom_estimator_list(
        self, binary_treatment_continuous_outcome_data
    ):
        """Test fitting with custom estimator list."""
        spec = NuisanceTunerSpec(
            fit_treatment_model=True,
            treatment_model_config={"estimator_list": ["lgbm"]},
        )
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_continuous_outcome_data, spec)

        assert tuner.treatment_model_ is not None

    def test_fit_with_multiple_configs(self, binary_treatment_continuous_outcome_data):
        """Test fitting with multiple model configs."""
        spec = NuisanceTunerSpec(
            fit_outcome_model=True,
            fit_regression_model=True,
            outcome_model_config={"estimator_list": ["rf"]},
            regression_model_config={"estimator_list": ["extra_tree"]},
        )
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_continuous_outcome_data, spec)

        assert tuner.outcome_model_ is not None
        assert tuner.regression_model_ is not None


# ==============================================================================
# TREATMENT TYPE TESTS
# ==============================================================================


class TestTreatmentTypes:
    """Test with different treatment types."""

    def test_fit_continuous_treatment(
        self, continuous_treatment_continuous_outcome_data
    ):
        """Test fitting with continuous treatment."""
        spec = NuisanceTunerSpec(fit_treatment_model=True)
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(continuous_treatment_continuous_outcome_data, spec)

        assert tuner.treatment_model_ is not None

    def test_fit_binary_outcome(self, binary_treatment_binary_outcome_data):
        """Test fitting with binary outcome."""
        spec = NuisanceTunerSpec(fit_outcome_model=True, fit_regression_model=True)
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(binary_treatment_binary_outcome_data, spec)

        assert tuner.outcome_model_ is not None
        assert tuner.regression_model_ is not None


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_fit_with_data_with_confounders(self):
        """Test fitting with dataset containing confounders."""
        gen = SyntheticDataGenerator(
            n_obs=100, n_cont_modifiers=2, n_cont_confounders=2, seed=42
        )

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            W=[c for c in gen.df.columns if "W" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        spec = NuisanceTunerSpec(fit_treatment_model=True, fit_outcome_model=True)
        tuner = NuisanceTuner(time_budget=1, seed=42, verbose=0)

        tuner.fit(data, spec)

        assert tuner.treatment_model_ is not None
        assert tuner.outcome_model_ is not None

    def test_base_config_structure(self):
        """Test that base config has expected structure."""
        tuner = NuisanceTuner(time_budget=60, seed=42)
        config = tuner._build_base_config()

        assert config["time_budget"] == 60
        assert config["seed"] == 42
        assert config["n_jobs"] == -1
        assert config["early_stop"] is True

    def test_ray_config_added(self):
        """Test that Ray config is properly added."""
        tuner = NuisanceTuner(use_ray=True)
        config = tuner._build_base_config()

        assert config["use_ray"] is True
        assert "n_concurrent_trials" in config

    def test_spark_config_added(self):
        """Test that Spark config is properly added."""
        tuner = NuisanceTuner(use_spark=True)
        config = tuner._build_base_config()

        assert config["use_spark"] is True
        assert "n_concurrent_trials" in config
