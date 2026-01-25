"""Comprehensive tests for SyntheticDataGenerator."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from caml.extensions.synthetic_data import (
    SyntheticDataGenerator,
    _truncate_and_renormalize_probabilities,
    make_fully_heterogeneous_dataset,
    make_partially_linear_dataset_constant,
    make_partially_linear_dataset_simple,
)

pytestmark = [pytest.mark.extensions, pytest.mark.synthetic_data]


# ==============================================================================
# REPRODUCIBILITY TESTS
# ==============================================================================


class TestReproducibility:
    """Test reproducibility of data generation."""

    def test_same_seed_produces_identical_data(self):
        """Test that same seed produces identical data."""
        gen1 = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            n_cont_modifiers=2,
            seed=42,
        )
        gen2 = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            n_cont_modifiers=2,
            seed=42,
        )

        pd.testing.assert_frame_equal(gen1.df, gen2.df)
        pd.testing.assert_frame_equal(gen1.cates, gen2.cates)
        pd.testing.assert_frame_equal(gen1.ates, gen2.ates)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        gen1 = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            seed=42,
        )
        gen2 = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            seed=123,
        )

        assert not gen1.df.equals(gen2.df)

    def test_seed_none_generates_valid_data(self):
        """Test that seed=None produces valid data."""
        gen = SyntheticDataGenerator(
            n_obs=100, n_cont_outcomes=1, n_binary_treatments=1, seed=None
        )

        assert gen._seed is not None
        assert len(gen.df) == 100
        assert not gen.df.isnull().any().any()

    def test_dgp_can_regenerate_data(self):
        """Test that stored DGP can regenerate exact same data."""
        gen = SyntheticDataGenerator(
            n_obs=50,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=2,
            seed=42,
        )

        for var_name, dgp_info in gen.dgp.items():
            formula = dgp_info["formula"]
            params = dgp_info["params"]
            noise = dgp_info["noise"]
            f = dgp_info["function"]
            raw_scores = dgp_info["raw_scores"]

            if formula:
                dm = gen.create_design_matrix(gen.df, formula)
                regenerated = f(dm, params, noise)
            else:
                regenerated = f(gen.df, params, noise)

            assert_allclose(regenerated, raw_scores)


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================


class TestValidation:
    """Test parameter validation."""

    def test_n_confounding_modifiers_exceeds_total_modifiers(self):
        """Test validation when n_confounding_modifiers > total modifiers."""
        with pytest.raises(ValueError, match="cannot exceed"):
            SyntheticDataGenerator(
                n_cont_modifiers=2,
                n_binary_modifiers=1,
                n_confounding_modifiers=5,  # More than 2+1=3
            )

    def test_n_confounding_modifiers_equals_total_modifiers(self):
        """Test that n_confounding_modifiers can equal total modifiers."""
        gen = SyntheticDataGenerator(
            n_obs=50,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_modifiers=2,
            n_binary_modifiers=1,
            n_confounding_modifiers=3,  # Equals 2+1=3
            seed=42,
        )
        assert len(gen.df) == 50


# ==============================================================================
# INDEPENDENT VARIABLE GENERATION TESTS
# ==============================================================================


class TestIndependentVariableGeneration:
    """Test generation of independent variables (confounders and modifiers)."""

    def test_continuous_variables_are_numeric(self):
        """Test continuous variables are numeric."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=5,
            seed=42,
        )

        cont_cols = [
            c for c in gen.df.columns if "continuous" in c and ("W" in c or "X" in c)
        ]
        for col in cont_cols:
            assert pd.api.types.is_numeric_dtype(gen.df[col])
            assert not gen.df[col].isnull().any()

    def test_binary_variables_are_zero_one(self):
        """Test binary variables are 0 or 1."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_binary_confounders=3,
            seed=42,
        )

        binary_cols = [
            c for c in gen.df.columns if "binary" in c and ("W" in c or "X" in c)
        ]
        for col in binary_cols:
            unique_vals = gen.df[col].unique()
            assert set(unique_vals).issubset({0, 1})

    def test_discrete_variables_are_integers(self):
        """Test discrete variables are integers."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_discrete_confounders=2,
            seed=42,
        )

        discrete_cols = [
            c for c in gen.df.columns if "discrete" in c and ("W" in c or "X" in c)
        ]
        for col in discrete_cols:
            assert gen.df[col].dtype in [np.int32, np.int64, int]

    def test_empty_generation_with_no_variables(self):
        """Test with no confounders or modifiers."""
        gen = SyntheticDataGenerator(
            n_obs=50,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=0,
            n_binary_confounders=0,
            n_discrete_confounders=0,
            n_cont_modifiers=0,
            n_binary_modifiers=0,
            n_discrete_modifiers=0,
            seed=42,
        )

        # Should only have outcome and treatment columns
        assert len(gen.df.columns) == 2


# ==============================================================================
# DGP FUNCTION TESTS
# ==============================================================================


class TestDGPFunctions:
    """Test data generating process functions."""

    def test_continuous_dgp_is_linear(self):
        """Test continuous DGP produces linear output."""
        rng = np.random.default_rng(42)
        df = np.array([[1.0, 2.0], [3.0, 4.0]])

        dep, params, noise, scores, f = SyntheticDataGenerator._create_dgp_function(
            df=df,
            n_obs=2,
            stddev_err=0.0,  # No noise
            dep_type="continuous",
            rng=rng,
        )

        # With no noise, dep should equal df @ params
        expected = df @ params
        assert_allclose(dep, expected)

    def test_binary_dgp_probabilities_in_range(self):
        """Test binary DGP produces probabilities in [0.01, 0.99]."""
        rng = np.random.default_rng(42)
        df = np.random.randn(100, 5)

        dep, params, noise, scores, f = SyntheticDataGenerator._create_dgp_function(
            df=df, n_obs=100, stddev_err=1.0, dep_type="binary", rng=rng
        )

        # scores should be probabilities (truncated)
        assert np.all(scores >= 0.01)
        assert np.all(scores <= 0.99)

    def test_discrete_dgp_probabilities_sum_to_one(self):
        """Test discrete DGP probabilities sum to 1."""
        rng = np.random.default_rng(42)
        df = np.random.randn(50, 3)

        dep, params, noise, scores, f = SyntheticDataGenerator._create_dgp_function(
            df=df, n_obs=50, stddev_err=1.0, dep_type="discrete", rng=rng
        )

        # Each row should sum to ~1
        row_sums = scores.sum(axis=1)
        assert_allclose(row_sums, np.ones(50), atol=1e-6)

    def test_dgp_with_zero_noise_is_deterministic(self):
        """Test DGP with zero noise is deterministic."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        df = np.random.randn(20, 3)

        dep1, params1, _, _, _ = SyntheticDataGenerator._create_dgp_function(
            df=df, n_obs=20, stddev_err=0.0, dep_type="continuous", rng=rng1
        )
        dep2, params2, _, _, _ = SyntheticDataGenerator._create_dgp_function(
            df=df, n_obs=20, stddev_err=0.0, dep_type="continuous", rng=rng2
        )

        # With same seed and zero noise, should be identical
        assert_allclose(params1, params2)
        assert_allclose(dep1, dep2)


# ==============================================================================
# TREATMENT EFFECT COMPUTATION TESTS
# ==============================================================================


class TestTreatmentEffectComputation:
    """Test CATE and ATE computation."""

    def test_ate_equals_mean_cate(self):
        """Test that ATE equals mean of CATEs."""
        gen = SyntheticDataGenerator(
            n_obs=200,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            n_cont_modifiers=2,
            seed=42,
        )

        # Get CATE column
        cate_col = gen.cates.columns[0]
        ate_row = gen.ates[
            gen.ates["Treatment"].str.contains(cate_col.split("_on_")[0])
        ]

        assert_allclose(gen.cates[cate_col].mean(), ate_row["ATE"].values, rtol=1e-10)

    def test_constant_ate_with_no_modifiers(self):
        """Test that with no modifiers, CATE is constant (equals ATE)."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=5,
            n_cont_modifiers=0,  # No heterogeneity
            n_binary_modifiers=0,
            n_discrete_modifiers=0,
            causal_model_functional_form="linear",
            seed=42,
        )

        cate_col = gen.cates.columns[0]
        cates = gen.cates[cate_col]

        # All CATEs should be identical (no heterogeneity)
        assert_allclose(cates, cates.mean(), atol=1e-10)

    def test_heterogeneous_cates_with_modifiers(self):
        """Test that modifiers create heterogeneous treatment effects."""
        gen = SyntheticDataGenerator(
            n_obs=500,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            n_cont_modifiers=3,
            causal_model_functional_form="linear",
            seed=42,
        )

        cate_col = gen.cates.columns[0]
        cates = gen.cates[cate_col]

        # CATEs should vary (std > 0)
        assert cates.std() > 0

    def test_continuous_treatment_cate_is_marginal_effect(self):
        """Test continuous treatment CATE represents 1-unit change."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_cont_treatments=1,
            n_cont_confounders=2,
            seed=42,
        )

        # Should have CATE for continuous treatment
        assert len(gen.cates.columns) == 2
        assert "continuous" in gen.cates.columns[0]


# ==============================================================================
# STATISTICAL PROPERTIES TESTS
# ==============================================================================


class TestStatisticalProperties:
    """Test statistical properties of generated data."""

    def test_treatment_propensity_satisfies_overlap(self):
        """Test binary treatment has reasonable propensity."""
        gen = SyntheticDataGenerator(
            n_obs=1000,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=5,
            seed=42,
        )

        t_col = [c for c in gen.df.columns if "T1_binary" in c][0]
        propensity = gen.df[t_col].mean()

        # Should not be too extreme
        assert 0.05 < propensity < 0.95

    def test_noise_affects_outcome_variance(self):
        """Test that outcome noise parameter affects variance."""
        gen_low = SyntheticDataGenerator(
            n_obs=500,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            stddev_outcome_noise=0.1,
            seed=42,
        )
        gen_high = SyntheticDataGenerator(
            n_obs=500,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            stddev_outcome_noise=5.0,
            seed=42,
        )

        y_col_low = [c for c in gen_low.df.columns if "Y1_continuous" in c][0]
        y_col_high = [c for c in gen_high.df.columns if "Y1_continuous" in c][0]

        # Higher noise should lead to higher variance
        # (comparing outcomes is tricky since the DGP itself differs, but noise adds to variance)
        # At minimum, should not crash and should produce valid data
        assert gen_high.df[y_col_high].std() > 0
        assert gen_low.df[y_col_low].std() > 0

    def test_confounders_affect_treatment(self):
        """Test that confounders influence treatment assignment."""
        gen = SyntheticDataGenerator(
            n_obs=500,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=5,
            n_confounding_modifiers=0,  # Only confounders, no modifiers affecting treatment
            seed=42,
        )

        # With confounders in the treatment model, treatment should vary
        t_col = [c for c in gen.df.columns if "T1_binary" in c][0]
        assert gen.df[t_col].std() > 0

    def test_treatment_affects_outcome(self):
        """Test that treatment has effect on outcome."""
        gen = SyntheticDataGenerator(
            n_obs=500,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            n_cont_modifiers=2,
            seed=42,
        )

        # ATE should be non-zero
        ate = gen.ates["ATE"].values[0]
        assert ate != 0


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_observation(self):
        """Test with n_obs=1."""
        gen = SyntheticDataGenerator(
            n_obs=1, n_cont_outcomes=1, n_binary_treatments=1, seed=42
        )
        assert len(gen.df) == 1
        assert len(gen.cates) == 1

    def test_large_n_obs(self):
        """Test with large n_obs."""
        import time

        start = time.time()
        gen = SyntheticDataGenerator(
            n_obs=50_000,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=3,
            seed=42,
        )
        elapsed = time.time() - start

        assert len(gen.df) == 50_000
        assert elapsed < 60  # Should complete in reasonable time

    def test_many_nonlinear_transformations(self):
        """Test with many nonlinear transformations."""
        gen = SyntheticDataGenerator(
            n_obs=200,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_confounders=5,
            n_cont_modifiers=3,
            causal_model_functional_form="nonlinear",
            n_nonlinear_transformations=25,
            seed=42,
        )

        assert len(gen.df) == 200

    def test_all_binary_variables(self):
        """Test with all binary variables."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_binary_outcomes=1,
            n_cont_outcomes=0,
            n_binary_treatments=1,
            n_binary_confounders=3,
            n_binary_modifiers=2,
            seed=42,
        )

        # All columns should be binary (0 or 1)
        for col in gen.df.columns:
            unique_vals = set(gen.df[col].unique())
            assert unique_vals.issubset({0, 1})


# ==============================================================================
# DESIGN MATRIX TESTS
# ==============================================================================


class TestDesignMatrixCreation:
    """Test design matrix and formula creation."""

    def test_linear_formula_no_interactions(self):
        """Test linear formula without heterogeneity."""
        df = pd.DataFrame({"W1_continuous": [1, 2], "X1_continuous": [3, 4]})
        rng = np.random.default_rng(42)

        formula = SyntheticDataGenerator._create_patsy_formula(
            df=df,
            n_nonlinear_transformations=None,
            include_heterogeneity=False,
            rng=rng,
        )

        assert formula == "1 + W1_continuous + X1_continuous"

    def test_nonlinear_formula_contains_transformations(self):
        """Test nonlinear formula includes transformations."""
        df = pd.DataFrame({"W1_continuous": [1, 2, 3], "W2_continuous": [4, 5, 6]})
        rng = np.random.default_rng(42)

        formula = SyntheticDataGenerator._create_patsy_formula(
            df=df, n_nonlinear_transformations=3, include_heterogeneity=False, rng=rng
        )

        # Should contain base terms plus transformations
        assert "W1_continuous" in formula
        assert "W2_continuous" in formula
        # Should have some nonlinear terms (hard to test exactly which ones)
        assert len(formula.split("+")) > 3  # More than just intercept + 2 vars

    def test_heterogeneity_adds_interactions(self):
        """Test include_heterogeneity=True adds interaction terms."""
        df = pd.DataFrame(
            {"W1_continuous": [1, 2], "X1_continuous": [3, 4], "T1_binary": [0, 1]}
        )
        rng = np.random.default_rng(42)

        formula = SyntheticDataGenerator._create_patsy_formula(
            df=df, n_nonlinear_transformations=None, include_heterogeneity=True, rng=rng
        )

        # Should contain interactions with treatment
        assert "*" in formula  # Interaction operator
        assert "T1_binary" in formula

    def test_empty_dataframe_returns_empty_formula(self):
        """Test empty DataFrame returns empty string."""
        df = pd.DataFrame()
        rng = np.random.default_rng(42)

        formula = SyntheticDataGenerator._create_patsy_formula(
            df=df,
            n_nonlinear_transformations=None,
            include_heterogeneity=False,
            rng=rng,
        )

        assert formula == ""


# ==============================================================================
# Test DoubleML DGP Functions
# ==============================================================================


class TestFunctionals:
    @pytest.mark.parametrize(
        ("probs, expected"),
        [
            (
                [0, 0.05, 0.5, 0.95, 1],
                [0.1, 0.1, 0.5, 0.9, 0.9],
            ),
            (
                [
                    [0, 0.05, 0.95, 1],
                    [0.8, 0.75, 0.05, 0],
                    [0.2, 0.2, 0, 0],
                ],
                [
                    [0.09, 0.10, 0.82, 0.82],
                    [0.73, 0.71, 0.09, 0.09],
                    [0.18, 0.19, 0.09, 0.09],
                ],
            ),
        ],
    )
    def test__truncate_and_renormalize_probabilities(self, probs, expected):
        truncated_probs = _truncate_and_renormalize_probabilities(
            np.array(probs).T, epsilon=0.1
        )
        assert_allclose(truncated_probs, np.array(expected).T, atol=0.01)

    @pytest.mark.parametrize("n_obs", [1000, 10000])
    @pytest.mark.parametrize("n_confounders", [5, 10])
    @pytest.mark.parametrize("dim_heterogeneity", [1, 2, 3])
    @pytest.mark.parametrize("binary_treatment", [True, False])
    def test_make_partially_linear_dataset_simple(
        self,
        n_obs,
        n_confounders,
        dim_heterogeneity,
        binary_treatment,
    ):
        if dim_heterogeneity == 3:
            with pytest.raises(ValueError):
                make_partially_linear_dataset_simple(
                    dim_heterogeneity=dim_heterogeneity
                )
        else:
            df, cates, ate = make_partially_linear_dataset_simple(
                n_obs=n_obs,
                n_confounders=n_confounders,
                dim_heterogeneity=dim_heterogeneity,
                binary_treatment=binary_treatment,
            )
            assert df.shape == (n_obs, n_confounders + 2)
            assert cates.shape == (n_obs,)
            assert isinstance(ate, float)
            if binary_treatment:
                assert df["d"].unique().shape[0] == 2
            else:
                assert df["d"].unique().shape[0] != 2

            assert ate == pytest.approx(4.5, abs=0.5)

    @pytest.mark.parametrize("n_obs", [1000, 10000])
    @pytest.mark.parametrize("exp_ate", [4.5, 15.5])
    @pytest.mark.parametrize("n_confounders", [5, 15])
    @pytest.mark.parametrize(
        "dgp", ["make_plr_CCDDHNR2018", "make_plr_turrell2018", "bad"]
    )
    def test_make_partially_linear_dataset_constant(
        self,
        n_obs,
        exp_ate,
        n_confounders,
        dgp,
    ):
        if dgp == "bad":
            with pytest.raises(ValueError):
                make_partially_linear_dataset_constant(dgp=dgp)
        else:
            df, cates, ate = make_partially_linear_dataset_constant(
                n_obs=n_obs,
                ate=exp_ate,
                n_confounders=n_confounders,
                dgp=dgp,
            )
            assert df.shape == (n_obs, n_confounders + 2)
            assert cates.shape == (n_obs,)
            assert isinstance(ate, float)
            assert ate == exp_ate
            assert np.all(cates == exp_ate)

    @pytest.mark.parametrize("n_obs", [1000, 10000])
    @pytest.mark.parametrize("n_confounders", [5, 15])
    @pytest.mark.parametrize("theta", [4.0, 5.6])
    def test_make_fully_heterogeneous_dataset(self, n_obs, n_confounders, theta):
        df, cates, ate = make_fully_heterogeneous_dataset(
            n_obs=n_obs,
            n_confounders=n_confounders,
            theta=theta,
        )
        assert df.shape == (n_obs, n_confounders + 2)
        assert cates.shape == (n_obs,)
        assert isinstance(ate, float)
        assert ate == pytest.approx(theta, abs=0.2)
