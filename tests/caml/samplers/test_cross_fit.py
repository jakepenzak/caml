"""Tests for caml.samplers.cross_fit module."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.samplers import CrossFitter
from caml.utilities.synthetic_data import SyntheticDataGenerator

pytestmark = [pytest.mark.samplers]


@pytest.fixture
def binary_treatment_data():
    """Create synthetic data with binary treatment."""
    gen = SyntheticDataGenerator(
        n_obs=200,
        n_cont_modifiers=3,
        n_cont_confounders=2,
        n_binary_treatments=1,
        n_cont_outcomes=1,
        seed=42,
    )
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        W=[c for c in gen.df.columns if "W" in c],
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def continuous_treatment_data():
    """Create synthetic data with continuous treatment."""
    gen = SyntheticDataGenerator(
        n_obs=200,
        n_cont_modifiers=3,
        n_cont_treatments=1,
        n_binary_treatments=0,
        n_cont_outcomes=1,
        seed=42,
    )
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_continuous",
        Y="Y1_continuous",
        treatment_type=TreatmentType.CONTINUOUS,
        outcome_type=OutcomeType.CONTINUOUS,
    )


class TestCrossFitterInit:
    """Test CrossFitter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        cf = CrossFitter()

        assert cf.cv == 3
        assert cf.random_state is None

    def test_custom_init(self):
        """Test custom initialization parameters."""
        cf = CrossFitter(cv=5, random_state=42)

        assert cf.cv == 5
        assert cf.random_state == 42


class TestFitPredictOutcomeModel:
    """Test fit_predict_outcome_model method."""

    def test_returns_correct_shape(self, binary_treatment_data):
        """Test that output has correct shape."""
        cf = CrossFitter(cv=3, random_state=42)
        outcome_model = LinearRegression()

        m_hat = cf.fit_predict_outcome_model(binary_treatment_data, outcome_model)

        assert m_hat.shape == (200, 1)

    def test_predictions_reasonable_range(self, binary_treatment_data):
        """Test that predictions are in reasonable range."""
        cf = CrossFitter(cv=3, random_state=42)
        outcome_model = LinearRegression()

        m_hat = cf.fit_predict_outcome_model(binary_treatment_data, outcome_model)

        # Predictions should not be all same value (model is actually fitting)
        assert np.std(m_hat) > 0


class TestFitPredictTreatmentModel:
    """Test fit_predict_treatment_model method."""

    def test_binary_treatment_returns_probabilities(self, binary_treatment_data):
        """Test that binary treatment returns probabilities."""
        cf = CrossFitter(cv=3, random_state=42)
        treatment_model = LogisticRegression()

        e_hat = cf.fit_predict_treatment_model(binary_treatment_data, treatment_model)

        assert e_hat.shape == (200, 1)
        # Probabilities should be between 0 and 1
        assert np.all(e_hat >= 0) and np.all(e_hat <= 1)

    def test_continuous_treatment_returns_predictions(self, continuous_treatment_data):
        """Test that continuous treatment returns predictions."""
        cf = CrossFitter(cv=3, random_state=42)
        treatment_model = LinearRegression()

        e_hat = cf.fit_predict_treatment_model(
            continuous_treatment_data, treatment_model
        )

        assert e_hat.shape == (200, 1)
        # Predictions should vary
        assert np.std(e_hat) > 0


class TestFitPredictRegressionModel:
    """Test fit_predict_regression_model method."""

    def test_returns_two_arrays(self, binary_treatment_data):
        """Test that method returns mu_0 and mu_1."""
        cf = CrossFitter(cv=3, random_state=42)
        regression_model = LinearRegression()

        mu_0, mu_1 = cf.fit_predict_regression_model(
            binary_treatment_data, regression_model
        )

        assert mu_0.shape == (200, 1)
        assert mu_1.shape == (200, 1)

    def test_mu_0_and_mu_1_differ(self, binary_treatment_data):
        """Test that control and treatment predictions differ."""
        cf = CrossFitter(cv=3, random_state=42)
        regression_model = LinearRegression()

        mu_0, mu_1 = cf.fit_predict_regression_model(
            binary_treatment_data, regression_model
        )

        # Models are trained on different subsets, predictions should differ
        assert not np.allclose(mu_0, mu_1)


class TestFitPredictNuisancesDML:
    """Test fit_predict_nuisances_dml method."""

    def test_returns_outcome_and_treatment_predictions(self, binary_treatment_data):
        """Test that method returns both m_hat and l_hat."""
        cf = CrossFitter(cv=3, random_state=42)

        m_hat, l_hat = cf.fit_predict_nuisances_dml(
            data=binary_treatment_data,
            outcome_model=LinearRegression(),
            treatment_model=LogisticRegression(),
        )

        assert m_hat.shape == (200, 1)
        assert l_hat.shape == (200, 1)


class TestFitPredictNuisancesDR:
    """Test fit_predict_nuisances_dr method."""

    def test_returns_three_arrays(self, binary_treatment_data):
        """Test that method returns mu_0, mu_1, and e_hat."""
        cf = CrossFitter(cv=3, random_state=42)

        mu_0, mu_1, e_hat = cf.fit_predict_nuisances_dr(
            data=binary_treatment_data,
            regression_model=LinearRegression(),
            treatment_model=LogisticRegression(),
        )

        assert mu_0.shape == (200, 1)
        assert mu_1.shape == (200, 1)
        assert e_hat.shape == (200, 1)

    def test_propensity_scores_valid(self, binary_treatment_data):
        """Test that propensity scores are valid probabilities."""
        cf = CrossFitter(cv=3, random_state=42)

        _, _, e_hat = cf.fit_predict_nuisances_dr(
            data=binary_treatment_data,
            regression_model=LinearRegression(),
            treatment_model=LogisticRegression(),
        )

        assert np.all(e_hat >= 0) and np.all(e_hat <= 1)


class TestCrossFitterDeterminism:
    """Test that CrossFitter produces deterministic results with random_state."""

    def test_same_random_state_same_results(self, binary_treatment_data):
        """Test that same random_state produces identical results."""
        cf1 = CrossFitter(cv=3, random_state=42)
        cf2 = CrossFitter(cv=3, random_state=42)

        m_hat1 = cf1.fit_predict_outcome_model(
            binary_treatment_data, LinearRegression()
        )
        m_hat2 = cf2.fit_predict_outcome_model(
            binary_treatment_data, LinearRegression()
        )

        np.testing.assert_array_almost_equal(m_hat1, m_hat2)


class TestCrossFitterGroundTruth:
    """Test CrossFitter predictions against ground truth from simulated data."""

    @pytest.fixture
    def linear_dgp_data(self):
        """Create simple linear DGP data with known ground truth."""
        gen = SyntheticDataGenerator(
            n_obs=1000,
            n_cont_modifiers=3,
            n_cont_confounders=2,
            n_binary_treatments=1,
            n_cont_outcomes=1,
            causal_model_functional_form="linear",
            seed=42,
        )
        return gen

    def test_propensity_predictions_correlate_with_truth(self, linear_dgp_data):
        """Test that treatment model predictions correlate with true propensities."""
        gen = linear_dgp_data
        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            W=[c for c in gen.df.columns if "W" in c],
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        # Get true propensity scores from DGP
        true_propensity = gen.dgp["T1_binary"]["raw_scores"]

        cf = CrossFitter(cv=5, random_state=42)
        e_hat = cf.fit_predict_treatment_model(data, LogisticRegression())

        # Correlation should be positive and reasonably strong
        correlation = np.corrcoef(true_propensity, e_hat.ravel())[0, 1]
        assert correlation > 0.3, f"Propensity correlation too low: {correlation:.3f}"

    def test_outcome_predictions_correlate_with_observations(self, linear_dgp_data):
        """Test that outcome model predictions correlate with observed outcomes."""
        gen = linear_dgp_data
        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            W=[c for c in gen.df.columns if "W" in c],
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        cf = CrossFitter(cv=5, random_state=42)
        m_hat = cf.fit_predict_outcome_model(data, LinearRegression())

        # Predictions should correlate with actual outcomes
        correlation = np.corrcoef(data.Y.ravel(), m_hat.ravel())[0, 1]
        assert correlation > 0.3, f"Outcome correlation too low: {correlation:.3f}"

    def test_regression_model_captures_treatment_heterogeneity(self, linear_dgp_data):
        """Test that mu_1 - mu_0 correlates with true CATEs."""
        gen = linear_dgp_data
        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            W=[c for c in gen.df.columns if "W" in c],
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        # Get true CATEs
        true_cates = gen.cates.iloc[:, 0].values

        cf = CrossFitter(cv=5, random_state=42)
        mu_0, mu_1 = cf.fit_predict_regression_model(data, LinearRegression())

        # Estimated CATE from regression models
        estimated_cate = (mu_1 - mu_0).ravel()

        # Correlation should be positive (model captures heterogeneity direction)
        correlation = np.corrcoef(true_cates, estimated_cate)[0, 1]
        assert correlation > 0.2, f"CATE correlation too low: {correlation:.3f}"
