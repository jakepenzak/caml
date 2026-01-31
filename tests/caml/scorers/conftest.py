"""Shared fixtures for scorer tests."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.estimators.dml import WrappedLinearDML
from caml.extensions.synthetic_data import SyntheticDataGenerator


@pytest.fixture(scope="module")
def synthetic_data_gen():
    """Create synthetic data generator with ground truth CATEs."""
    return SyntheticDataGenerator(
        n_obs=500,
        n_cont_modifiers=3,
        n_cont_confounders=2,
        n_binary_treatments=1,
        n_cont_outcomes=1,
        causal_model_functional_form="linear",
        seed=42,
    )


@pytest.fixture(scope="module")
def causal_dataset(synthetic_data_gen):
    """Create CausalDataset with true CATEs."""
    gen = synthetic_data_gen
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        W=[c for c in gen.df.columns if "W" in c],
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
        true_cates=np.array(gen.cates.iloc[:, 0]).reshape(-1, 1),
    )


@pytest.fixture(scope="module")
def fitted_estimator(causal_dataset):
    """Create and fit a LinearDML estimator."""
    estimator = WrappedLinearDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(),
        cv=3,
        random_state=42,
    )
    estimator.fit(causal_dataset)
    return estimator
