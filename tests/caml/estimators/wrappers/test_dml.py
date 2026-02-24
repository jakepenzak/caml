"""Tests for DML wrapper estimators."""

import numpy as np
import pytest
from caml.estimators.dml import (
    WrappedCausalForestDML,
    WrappedKernelDML,
    WrappedLinearDML,
    WrappedNonParamDML,
    WrappedSparseLinearDML,
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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


class TestWrappedLinearDML:
    """Tests for WrappedLinearDML."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedLinearDML.capabilities is not None
        assert TreatmentType.BINARY in WrappedLinearDML.capabilities.treatment_types
        assert OutcomeType.CONTINUOUS in WrappedLinearDML.capabilities.outcome_types

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = WrappedLinearDML(
            model_y=LinearRegression(), model_t=LogisticRegression()
        )
        assert estimator._is_fitted is False
        assert hasattr(estimator, "_estimator")

    def test_protocol_compliance(self, binary_continuous_data):
        """Test that estimator implements AutoCateEstimator protocol."""
        estimator = WrappedLinearDML(
            model_y=LinearRegression(), model_t=LogisticRegression()
        )
        estimator.fit(binary_continuous_data)
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedLinearDML(
            model_y=LinearRegression(), model_t=LogisticRegression(), random_state=42
        )
        estimator.fit(binary_continuous_data)

        # Check fitted flag
        assert estimator._is_fitted is True

        # Predict CATE
        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()

    def test_compatibility_check(self, binary_continuous_data):
        """Test compatibility checking."""
        estimator = WrappedLinearDML(
            model_y=LinearRegression(), model_t=LogisticRegression()
        )
        # Should not raise
        assert estimator.is_compatible_with(binary_continuous_data)

    def test_is_compatible_with_classmethod(self, binary_continuous_data):
        """Test classmethod is_compatible_with works without instantiation."""
        # This should not raise (no instantiation needed)
        assert WrappedLinearDML.is_compatible_with(binary_continuous_data)


class TestWrappedSparseLinearDML:
    """Tests for WrappedSparseLinearDML."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedSparseLinearDML.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedSparseLinearDML(
            model_y=LinearRegression(), model_t=LogisticRegression(), random_state=42
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()


class TestWrappedCausalForestDML:
    """Tests for WrappedCausalForestDML."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedCausalForestDML.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        # n_estimators must be divisible by subforest_size (default=4)
        estimator = WrappedCausalForestDML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(),
            n_estimators=100,
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()


class TestWrappedNonParamDML:
    """Tests for WrappedNonParamDML."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedNonParamDML.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedNonParamDML(
            model_y=GradientBoostingRegressor(n_estimators=50),
            model_t=LogisticRegression(),
            model_final=GradientBoostingRegressor(n_estimators=50),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()


class TestWrappedKernelDML:
    """Tests for WrappedKernelDML."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedKernelDML.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedKernelDML(
            model_y=RandomForestRegressor(n_estimators=50, random_state=42),
            model_t=LogisticRegression(random_state=42),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X), 1)
        assert np.isfinite(cate).all()
