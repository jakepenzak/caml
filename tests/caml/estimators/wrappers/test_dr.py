"""Tests for Doubly Robust wrapper estimators."""

import numpy as np
import pytest
from caml.estimators.dr import (
    WrappedDRLearner,
    WrappedForestDRLearner,
    WrappedLinearDRLearner,
    WrappedSparseLinearDRLearner,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.estimators import AutoCateEstimator
from caml.utilities.synthetic_data import SyntheticDataGenerator

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


class TestWrappedDRLearner:
    """Tests for WrappedDRLearner."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedDRLearner.capabilities is not None
        assert TreatmentType.BINARY in WrappedDRLearner.capabilities.treatment_types
        assert OutcomeType.CONTINUOUS in WrappedDRLearner.capabilities.outcome_types

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = WrappedDRLearner(
            model_propensity=LogisticRegression(),
            model_regression=GradientBoostingRegressor(n_estimators=50),
            model_final=GradientBoostingRegressor(n_estimators=50),
        )
        assert estimator._is_fitted is False
        assert hasattr(estimator, "_estimator")

    def test_protocol_compliance(self, binary_continuous_data):
        """Test that estimator implements AutoCateEstimator protocol."""
        estimator = WrappedDRLearner(
            model_propensity=LogisticRegression(),
            model_regression=GradientBoostingRegressor(n_estimators=50),
            model_final=GradientBoostingRegressor(n_estimators=50),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedDRLearner(
            model_propensity=LogisticRegression(),
            model_regression=GradientBoostingRegressor(n_estimators=50),
            model_final=GradientBoostingRegressor(n_estimators=50),
            random_state=42,
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
        assert WrappedDRLearner.is_compatible_with(binary_continuous_data)


class TestWrappedLinearDRLearner:
    """Tests for WrappedLinearDRLearner."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedLinearDRLearner.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedLinearDRLearner(
            model_propensity=LogisticRegression(),
            model_regression=LinearRegression(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()


class TestWrappedSparseLinearDRLearner:
    """Tests for WrappedSparseLinearDRLearner."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedSparseLinearDRLearner.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedSparseLinearDRLearner(
            model_propensity=LogisticRegression(),
            model_regression=LinearRegression(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()


class TestWrappedForestDRLearner:
    """Tests for WrappedForestDRLearner."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedForestDRLearner.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        # n_estimators must be divisible by subforest_size (default=4)
        estimator = WrappedForestDRLearner(
            model_propensity=GradientBoostingClassifier(n_estimators=50),
            model_regression=GradientBoostingRegressor(n_estimators=50),
            n_estimators=100,
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()
