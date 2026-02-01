"""Tests for Meta-learner wrapper estimators."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.estimators import AutoCateEstimator
from caml.estimators.meta import (
    WrappedSLearner,
    WrappedTLearner,
    WrappedXLearner,
)
from caml.extensions.synthetic_data import SyntheticDataGenerator

pytestmark = pytest.mark.estimators


@pytest.fixture
def binary_continuous_data():
    """Generate binary treatment, continuous outcome data."""
    gen = SyntheticDataGenerator(n_cont_modifiers=3, n_obs=500, seed=42)
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )


class TestWrappedSLearner:
    """Tests for WrappedSLearner."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedSLearner.capabilities is not None
        assert TreatmentType.BINARY in WrappedSLearner.capabilities.treatment_types
        assert OutcomeType.CONTINUOUS in WrappedSLearner.capabilities.outcome_types

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = WrappedSLearner(overall_model=GradientBoostingRegressor())
        assert estimator._is_fitted is False
        assert hasattr(estimator, "_estimator")

    def test_protocol_compliance(self, binary_continuous_data):
        """Test that estimator implements AutoCateEstimator protocol."""
        estimator = WrappedSLearner(
            overall_model=GradientBoostingRegressor(n_estimators=50, random_state=42)
        )
        estimator.fit(binary_continuous_data)
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedSLearner(
            overall_model=GradientBoostingRegressor(n_estimators=50, random_state=42)
        )
        estimator.fit(binary_continuous_data)

        # Check fitted flag
        assert estimator._is_fitted is True

        # Predict CATE
        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()

    def test_is_compatible_with_classmethod(self, binary_continuous_data):
        """Test classmethod is_compatible_with works without instantiation."""
        assert WrappedSLearner.is_compatible_with(binary_continuous_data)


class TestWrappedTLearner:
    """Tests for WrappedTLearner."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedTLearner.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedTLearner(
            models=GradientBoostingRegressor(n_estimators=50, random_state=42)
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()


class TestWrappedXLearner:
    """Tests for WrappedXLearner."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedXLearner.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedXLearner(
            models=GradientBoostingRegressor(n_estimators=50, random_state=42),
            cate_models=GradientBoostingRegressor(n_estimators=50, random_state=42),
            propensity_model=LogisticRegression(random_state=42),
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()
