"""Tests for Orthogonal Random Forest wrapper estimators."""

import numpy as np
import pytest
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.estimators import AutoCateEstimator
from caml.estimators.wrappers.orf import WrappedDMLOrthoForest, WrappedDROrthoForest
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


class TestWrappedDMLOrthoForest:
    """Tests for WrappedDMLOrthoForest."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedDMLOrthoForest.clean_name == "DMLOrthoForest"
        assert WrappedDMLOrthoForest.capabilities is not None
        assert (
            TreatmentType.BINARY in WrappedDMLOrthoForest.capabilities.treatment_types
        )
        assert (
            OutcomeType.CONTINUOUS in WrappedDMLOrthoForest.capabilities.outcome_types
        )

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = WrappedDMLOrthoForest(
            n_trees=50, model_T=LogisticRegressionCV(), model_Y=LassoCV()
        )
        assert estimator._is_fitted is False
        assert hasattr(estimator, "_estimator")

    def test_protocol_compliance(self, binary_continuous_data):
        """Test that estimator implements AutoCateEstimator protocol."""
        estimator = WrappedDMLOrthoForest(
            n_trees=50,
            model_T=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)
        assert isinstance(estimator, AutoCateEstimator)

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedDMLOrthoForest(
            n_trees=50,
            max_depth=30,
            model_T=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        # Check fitted flag
        assert estimator._is_fitted is True

        # Predict CATE
        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X),)
        assert np.isfinite(cate).all()

    def test_is_compatible_with_classmethod(self, binary_continuous_data):
        """Test classmethod is_compatible_with works without instantiation."""
        assert WrappedDMLOrthoForest.is_compatible_with(binary_continuous_data)


class TestWrappedDROrthoForest:
    """Tests for WrappedDROrthoForest."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        assert WrappedDROrthoForest.clean_name == "DROrthoForest"
        assert WrappedDROrthoForest.capabilities is not None

    def test_fit_and_effect(self, binary_continuous_data):
        """Test fitting and CATE prediction."""
        estimator = WrappedDROrthoForest(
            n_trees=50,
            max_depth=30,
            propensity_model=LogisticRegressionCV(),
            model_Y=LassoCV(),
            random_state=42,
        )
        estimator.fit(binary_continuous_data)

        cate = estimator.effect(binary_continuous_data.X)
        assert cate.shape == (len(binary_continuous_data.X),)
        assert np.isfinite(cate).all()
