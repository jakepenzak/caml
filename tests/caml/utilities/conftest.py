"""Shared fixtures for synthetic data and extensions testing."""

import pytest

from caml.utilities.synthetic_data import SyntheticDataGenerator

# ============================================================================
# STANDARD TEST SCENARIOS
# ============================================================================


@pytest.fixture(scope="session")
def simple_binary_treatment_dataset():
    """Simple dataset: binary treatment, continuous outcome, 5 confounders, 2 modifiers."""
    return SyntheticDataGenerator(
        n_obs=1000,
        n_cont_outcomes=1,
        n_binary_treatments=1,
        n_cont_confounders=5,
        n_cont_modifiers=2,
        n_confounding_modifiers=1,
        stddev_outcome_noise=1.0,
        stddev_treatment_noise=1.0,
        causal_model_functional_form="linear",
        seed=42,
    )


@pytest.fixture(scope="session")
def nonlinear_dataset():
    """Nonlinear DGP with heterogeneity."""
    return SyntheticDataGenerator(
        n_obs=2000,
        n_cont_outcomes=1,
        n_binary_treatments=1,
        n_cont_confounders=10,
        n_cont_modifiers=3,
        causal_model_functional_form="nonlinear",
        n_nonlinear_transformations=15,
        seed=123,
    )


@pytest.fixture(scope="session")
def multi_treatment_dataset():
    """Dataset with multiple treatment types."""
    return SyntheticDataGenerator(
        n_obs=1500,
        n_cont_outcomes=1,
        n_cont_treatments=1,
        n_binary_treatments=1,
        n_discrete_treatments=1,
        n_cont_confounders=5,
        n_cont_modifiers=2,
        seed=789,
    )


@pytest.fixture(scope="session")
def constant_ate_dataset():
    """Dataset with no heterogeneity (constant ATE)."""
    return SyntheticDataGenerator(
        n_obs=1000,
        n_cont_outcomes=1,
        n_binary_treatments=1,
        n_cont_confounders=10,
        n_cont_modifiers=0,  # No modifiers = constant effect
        causal_model_functional_form="linear",
        seed=456,
    )


# ============================================================================
# PARAMETERIZED FIXTURES
# ============================================================================


@pytest.fixture(params=[42, 123, 789])
def seeded_generator(request):
    """Fixture providing generators with different seeds."""
    return SyntheticDataGenerator(
        n_obs=500,
        n_cont_outcomes=1,
        n_binary_treatments=1,
        n_cont_confounders=5,
        n_cont_modifiers=2,
        seed=request.param,
    )


@pytest.fixture(params=["linear", "nonlinear"])
def functional_form_generator(request):
    """Fixture providing both linear and nonlinear generators."""
    return SyntheticDataGenerator(
        n_obs=1000,
        n_cont_outcomes=1,
        n_binary_treatments=1,
        n_cont_confounders=5,
        n_cont_modifiers=2,
        causal_model_functional_form=request.param,
        seed=42,
    )


# ============================================================================
# EXTRACTED DATA FIXTURES
# ============================================================================


@pytest.fixture
def simple_causal_dataset(simple_binary_treatment_dataset):
    """Convert to CausalDataset for protocol testing."""
    from caml.data import CausalDataset, OutcomeType, TreatmentType

    gen = simple_binary_treatment_dataset

    X_cols = [c for c in gen.df.columns if c.startswith("X")]
    W_cols = [c for c in gen.df.columns if c.startswith("W")]
    T_col = [c for c in gen.df.columns if "T1_binary" in c][0]
    Y_col = [c for c in gen.df.columns if "Y1_continuous" in c][0]

    return CausalDataset.from_dataframe(
        gen.df,
        X=X_cols,
        T=T_col,
        Y=Y_col,
        W=W_cols,
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def known_cates(simple_binary_treatment_dataset):
    """Provide ground truth CATEs for validation."""
    return simple_binary_treatment_dataset.cates


@pytest.fixture
def known_ate(simple_binary_treatment_dataset):
    """Provide ground truth ATE for validation."""
    return simple_binary_treatment_dataset.ates
