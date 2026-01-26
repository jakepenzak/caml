"""Shared fixtures for estimator tests."""

import pytest

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.extensions.synthetic_data import SyntheticDataGenerator


@pytest.fixture
def binary_continuous_data():
    """Binary treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(
        n_cont_modifiers=3, n_binary_modifiers=1, n_obs=200, seed=42
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
def continuous_continuous_data():
    """Continuous treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(n_cont_modifiers=3, n_obs=200, seed=42)
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T2_continuous",
        Y="Y1_continuous",
        treatment_type=TreatmentType.CONTINUOUS,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def binary_binary_data():
    """Binary treatment, binary outcome dataset."""
    gen = SyntheticDataGenerator(n_cont_modifiers=3, n_obs=200, seed=42)
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y2_binary",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.BINARY,
    )


@pytest.fixture
def multi_continuous_data():
    """Multi-valued treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(n_cont_modifiers=3, n_obs=200, seed=42)
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T3_discrete",
        Y="Y1_continuous",
        treatment_type=TreatmentType.MULTI,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def small_binary_continuous_data():
    """Small dataset for quick tests - binary treatment, continuous outcome."""
    gen = SyntheticDataGenerator(n_cont_modifiers=2, n_obs=50, seed=123)
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def high_dimensional_data():
    """High-dimensional dataset for testing sparse methods."""
    gen = SyntheticDataGenerator(n_cont_modifiers=10, n_obs=150, seed=42)
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )
