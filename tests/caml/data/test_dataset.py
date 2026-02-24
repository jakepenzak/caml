"""Tests for caml.data.dataset module."""

import numpy as np
import pandas as pd
import pytest

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.utilities.synthetic_data import SyntheticDataGenerator

pytestmark = [pytest.mark.data]


# ==============================================================================
# DATASET CREATION TESTS
# ==============================================================================


class TestDatasetCreation:
    """Test CausalDataset creation and initialization."""

    def test_create_from_arrays(self):
        """Test creation from numpy arrays."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        T = np.random.binomial(1, 0.5, 100)
        Y = np.random.randn(100)

        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert len(data.Y) == 100
        assert data.treatment_type == TreatmentType.BINARY
        assert data.outcome_type == OutcomeType.CONTINUOUS

    def test_create_from_dataframe_components(self):
        """Test creation from pandas DataFrames and Series."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["x1", "x2", "x3"])
        T = pd.Series(np.random.binomial(1, 0.5, 100), name="treatment")
        Y = pd.Series(np.random.randn(100), name="outcome")

        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert len(data.Y) == 100
        assert isinstance(data.X, np.ndarray)
        assert isinstance(data.T, np.ndarray)

    def test_create_with_confounders(self):
        """Test creation with additional confounders W."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        T = np.random.binomial(1, 0.5, 100)
        Y = np.random.randn(100)
        W = np.random.randn(100, 2)

        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            W=W,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert len(data.W) == 100

    def test_create_with_weights(self):
        """Test creation with sample weights."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        T = np.random.binomial(1, 0.5, 100)
        Y = np.random.randn(100)
        weights = np.random.uniform(0.5, 1.5, 100)

        data = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            weights=weights,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert len(data.weights) == 100

    def test_default_treatment_type_is_binary(self):
        """Test that default treatment type is BINARY."""
        gen = SyntheticDataGenerator(
            n_obs=50, n_cont_outcomes=1, n_binary_treatments=1, seed=42
        )

        data = CausalDataset(
            X=gen.df[[c for c in gen.df.columns if "X" in c or "W" in c]],
            T=gen.df["T1_binary"],
            Y=gen.df["Y1_continuous"],
        )

        assert data.treatment_type == TreatmentType.BINARY

    def test_default_outcome_type_is_continuous(self):
        """Test that default outcome type is CONTINUOUS."""
        gen = SyntheticDataGenerator(
            n_obs=50, n_cont_outcomes=1, n_binary_treatments=1, seed=42
        )

        data = CausalDataset(
            X=gen.df[[c for c in gen.df.columns if "X" in c or "W" in c]],
            T=gen.df["T1_binary"],
            Y=gen.df["Y1_continuous"],
        )

        assert data.outcome_type == OutcomeType.CONTINUOUS


# ==============================================================================
# FROM_DATAFRAME METHOD TESTS
# ==============================================================================


class TestFromDataframe:
    """Test CausalDataset.from_dataframe() method."""

    def test_from_dataframe_basic(self):
        """Test basic from_dataframe functionality."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_modifiers=3,
            seed=42,
        )

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert len(data.Y) == 100
        assert data.X_names == [c for c in gen.df.columns if "X" in c]
        assert data.T_name == "T1_binary"
        assert data.Y_name == "Y1_continuous"

    def test_from_dataframe_with_confounders(self):
        """Test from_dataframe with confounders."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_modifiers=3,
            n_cont_confounders=2,
            seed=42,
        )

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            W=[c for c in gen.df.columns if "W" in c],
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert data.W_names == [c for c in gen.df.columns if "W" in c]
        assert data.W.shape[1] == 2

    def test_from_dataframe_stores_column_names(self):
        """Test that from_dataframe stores column names correctly."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "income": [50000, 60000, 70000],
                "treated": [0, 1, 0],
                "outcome": [100, 150, 120],
            }
        )

        data = CausalDataset.from_dataframe(
            df,
            X=["age", "income"],
            T="treated",
            Y="outcome",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert data.X_names == ["age", "income"]
        assert data.T_name == "treated"
        assert data.Y_name == "outcome"

    def test_from_dataframe_missing_column_raises_error(self):
        """Test that missing column raises KeyError."""
        df = pd.DataFrame({"x1": [1, 2], "y": [3, 4]})

        with pytest.raises(KeyError):
            CausalDataset.from_dataframe(
                df,
                X=["x1"],
                T="missing_treatment",  # This column doesn't exist
                Y="y",
            )


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================


class TestValidation:
    """Test CausalDataset validation logic."""

    def test_mismatched_shapes_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        X = np.random.randn(100, 3)
        T = np.random.randn(50)  # Wrong length
        Y = np.random.randn(100)

        with pytest.raises(ValueError, match="must have length"):
            CausalDataset(X=X, T=T, Y=Y)

    def test_multi_dimensional_treatment_raises_error(self):
        """Test that multi-dimensional treatment raises error."""
        X = np.random.randn(100, 3)
        T = np.random.randn(100, 2)  # 2D
        Y = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 1-dimensional"):
            CausalDataset(X=X, T=T, Y=Y)

    def test_multi_dimensional_outcome_raises_error(self):
        """Test that multi-dimensional outcome raises error."""
        X = np.random.randn(100, 3)
        T = np.random.randn(100)
        Y = np.random.randn(100, 2)  # 2D

        with pytest.raises(ValueError, match="must be 1-dimensional"):
            CausalDataset(X=X, T=T, Y=Y)

    def test_binary_treatment_with_three_values_raises_error(self):
        """Test that binary treatment with 3 values raises error."""
        X = np.random.randn(100, 3)
        T = np.random.choice([0, 1, 2], 100)  # 3 values
        Y = np.random.randn(100)

        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            CausalDataset(X=X, T=T, Y=Y, treatment_type=TreatmentType.BINARY)

    def test_multi_treatment_with_two_values_raises_error(self):
        """Test that multi treatment with 2 values raises error."""
        X = np.random.randn(100, 3)
        T = np.random.choice([0, 1], 100)  # Only 2 values
        Y = np.random.randn(100)

        with pytest.raises(ValueError, match="MULTI expects more than 2 unique values"):
            CausalDataset(X=X, T=T, Y=Y, treatment_type=TreatmentType.MULTI)

    def test_binary_outcome_with_three_values_raises_error(self):
        """Test that binary outcome with 3 values raises error."""
        X = np.random.randn(100, 3)
        T = np.random.choice([0, 1], 100)
        Y = np.random.choice([0, 1, 2], 100)  # 3 values

        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            CausalDataset(X=X, T=T, Y=Y, outcome_type=OutcomeType.BINARY)

    def test_validation_called_on_init(self):
        """Test that validation is automatically called on initialization."""
        # This should raise during __post_init__
        with pytest.raises(ValueError):
            CausalDataset(
                X=np.random.randn(100, 3),
                T=np.random.randn(50),  # Wrong length
                Y=np.random.randn(100),
            )


# ==============================================================================
# TREATMENT AND OUTCOME TYPE TESTS
# ==============================================================================


class TestTreatmentAndOutcomeTypes:
    """Test different treatment and outcome type combinations."""

    def test_binary_treatment_continuous_outcome(self):
        """Test binary treatment with continuous outcome."""
        gen = SyntheticDataGenerator(
            n_obs=100, n_cont_outcomes=1, n_binary_treatments=1, seed=42
        )

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c or "W" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert data.treatment_type == TreatmentType.BINARY
        assert data.outcome_type == OutcomeType.CONTINUOUS

    def test_binary_treatment_binary_outcome(self):
        """Test binary treatment with binary outcome."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_binary_outcomes=1,
            n_cont_outcomes=0,
            n_binary_treatments=1,
            seed=42,
        )

        # Get the actual column names
        y_col = [c for c in gen.df.columns if "Y" in c and "binary" in c][0]
        t_col = [c for c in gen.df.columns if "T" in c and "binary" in c][0]

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c or "W" in c],
            T=t_col,
            Y=y_col,
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.BINARY,
        )

        assert data.treatment_type == TreatmentType.BINARY
        assert data.outcome_type == OutcomeType.BINARY

    def test_continuous_treatment_continuous_outcome(self):
        """Test continuous treatment with continuous outcome."""
        gen = SyntheticDataGenerator(
            n_obs=100, n_cont_outcomes=1, n_cont_treatments=1, seed=42
        )

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c or "W" in c],
            T="T1_continuous",
            Y="Y1_continuous",
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert data.treatment_type == TreatmentType.CONTINUOUS
        assert data.outcome_type == OutcomeType.CONTINUOUS

    def test_multi_treatment_continuous_outcome(self):
        """Test multi-valued treatment with continuous outcome."""
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_outcomes=1,
            n_discrete_treatments=1,
            n_binary_treatments=0,
            seed=42,
        )

        # Get the actual column names
        y_col = [c for c in gen.df.columns if "Y" in c and "continuous" in c][0]
        t_col = [c for c in gen.df.columns if "T" in c and "discrete" in c][0]

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c or "W" in c],
            T=t_col,
            Y=y_col,
            treatment_type=TreatmentType.MULTI,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert data.treatment_type == TreatmentType.MULTI
        assert data.outcome_type == OutcomeType.CONTINUOUS


# ==============================================================================
# METADATA TESTS
# ==============================================================================


class TestMetadata:
    """Test metadata storage and retrieval."""

    def test_feature_names_stored_correctly(self):
        """Test that feature names are stored correctly."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "treatment": [0, 1, 0],
                "outcome": [10, 20, 15],
            }
        )

        data = CausalDataset.from_dataframe(
            df, X=["feature1", "feature2"], T="treatment", Y="outcome"
        )

        assert data.X_names == ["feature1", "feature2"]
        assert data.T_name == "treatment"
        assert data.Y_name == "outcome"

    def test_default_names_when_created_from_arrays(self):
        """Test default names when created from arrays."""
        X = np.random.randn(50, 3)
        T = np.random.binomial(1, 0.5, 50)
        Y = np.random.randn(50)

        data = CausalDataset(X=X, T=T, Y=Y)

        assert data.X_names is None
        assert data.T_name == "treatment"
        assert data.Y_name == "outcome"

    def test_confounder_names_stored(self):
        """Test that confounder names are stored."""
        df = pd.DataFrame(
            {
                "x1": [1, 2, 1],
                "w1": [3, 4, 2],
                "w2": [5, 6, 6],
                "t": [0, 1, 0],
                "y": [10, 20, 30],
            }
        )

        data = CausalDataset.from_dataframe(df, X=["x1"], T="t", Y="y", W=["w1", "w2"])

        assert data.W_names == ["w1", "w2"]


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_observation(self):
        """Test with a single observation."""
        X = np.array([[1.0, 2.0]])
        T = np.array([1])
        Y = np.array([5.0])

        with pytest.raises(
            ValueError,
            match="expects numeric outcome values with more than 2 unique values",
        ):
            CausalDataset(X=X, T=T, Y=Y)

    def test_single_feature(self):
        """Test with a single feature."""
        X = np.random.randn(100, 1)
        T = np.random.binomial(1, 0.5, 100)
        Y = np.random.randn(100)

        data = CausalDataset(X=X, T=T, Y=Y)

        assert data.X.shape[1] == 1

    def test_no_confounders(self):
        """Test that W=None works correctly."""
        X = np.random.randn(100, 3)
        T = np.random.binomial(1, 0.5, 100)
        Y = np.random.randn(100)

        data = CausalDataset(X=X, T=T, Y=Y, W=None)

        assert data.W is None
        assert data.W_names is None

    def test_large_dataset(self):
        """Test with large dataset."""
        gen = SyntheticDataGenerator(
            n_obs=10_000,
            n_cont_outcomes=1,
            n_binary_treatments=1,
            n_cont_modifiers=10,
            seed=42,
        )

        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
        )

        assert len(data.Y) == 10_000
        assert data.X.shape[1] == 10


# ==============================================================================
# __POST_INIT__ TESTS
# ==============================================================================


class TestPostInit:
    """Test CausalDataset.__post_init__ behavior."""

    # ------------------------------------------------------------------
    # String-to-enum coercion
    # ------------------------------------------------------------------

    def test_string_treatment_type_converted_to_enum(self):
        """Test that a string treatment_type is coerced to TreatmentType enum."""
        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type="binary",
        )
        assert data.treatment_type is TreatmentType.BINARY
        assert isinstance(data.treatment_type, TreatmentType)

    def test_string_outcome_type_converted_to_enum(self):
        """Test that a string outcome_type is coerced to OutcomeType enum."""
        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            outcome_type="continuous",
        )
        assert data.outcome_type is OutcomeType.CONTINUOUS
        assert isinstance(data.outcome_type, OutcomeType)

    def test_enum_treatment_type_passed_through_unchanged(self):
        """Test that an enum treatment_type is not re-wrapped."""
        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
        )
        assert data.treatment_type is TreatmentType.BINARY

    def test_invalid_string_treatment_type_raises(self):
        """Test that an unrecognised string treatment_type raises ValueError."""
        with pytest.raises(ValueError):
            CausalDataset(
                X=np.random.randn(100, 3),
                T=np.random.binomial(1, 0.5, 100),
                Y=np.random.randn(100),
                treatment_type="not_a_type",
            )

    def test_invalid_string_outcome_type_raises(self):
        """Test that an unrecognised string outcome_type raises ValueError."""
        with pytest.raises(ValueError):
            CausalDataset(
                X=np.random.randn(100, 3),
                T=np.random.binomial(1, 0.5, 100),
                Y=np.random.randn(100),
                outcome_type="not_a_type",
            )

    # ------------------------------------------------------------------
    # arr_at_least_2d promotion
    # ------------------------------------------------------------------

    def test_1d_X_promoted_to_column_vector(self):
        """Test that 1-D X (shape (n,)) is promoted to (n, 1)."""
        np.random.seed(0)
        X_1d = np.random.randn(100)
        data = CausalDataset(
            X=X_1d,
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
        )
        assert data.X.ndim == 2
        assert data.X.shape == (100, 1)

    def test_2d_X_shape_unchanged(self):
        """Test that 2-D X is kept as-is."""
        np.random.seed(0)
        X_2d = np.random.randn(100, 5)
        data = CausalDataset(
            X=X_2d,
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
        )
        assert data.X.shape == (100, 5)

    def test_1d_T_promoted_to_column_vector(self):
        """Test that 1-D T (shape (n,)) is promoted to (n, 1)."""
        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
        )
        assert data.T.ndim == 2
        assert data.T.shape == (100, 1)

    def test_1d_Y_promoted_to_column_vector(self):
        """Test that 1-D Y (shape (n,)) is promoted to (n, 1)."""
        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
        )
        assert data.Y.ndim == 2
        assert data.Y.shape == (100, 1)

    def test_1d_W_promoted_to_column_vector(self):
        """Test that 1-D W (shape (n,)) is promoted to (n, 1)."""
        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            W=np.random.randn(100),
        )
        assert data.W.ndim == 2
        assert data.W.shape == (100, 1)

    def test_W_none_stays_none(self):
        """Test that W=None is not promoted and remains None."""
        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            W=None,
        )
        assert data.W is None

    def test_all_arrays_are_numpy_after_init(self):
        """Test that pandas inputs are converted to numpy arrays by post_init."""
        np.random.seed(0)
        data = CausalDataset(
            X=pd.DataFrame(np.random.randn(100, 3)),
            T=pd.Series(np.random.binomial(1, 0.5, 100)),
            Y=pd.Series(np.random.randn(100)),
        )
        assert isinstance(data.X, np.ndarray)
        assert isinstance(data.T, np.ndarray)
        assert isinstance(data.Y, np.ndarray)

    # ------------------------------------------------------------------
    # Validation is triggered
    # ------------------------------------------------------------------

    def test_validation_triggered_on_init(self):
        """Test that __post_init__ triggers validation; invalid data raises ValueError."""
        with pytest.raises(ValueError):
            CausalDataset(
                X=np.random.randn(100, 3),
                T=np.random.randn(80),  # length mismatch
                Y=np.random.randn(100),
            )


# ==============================================================================
# SAMPLE METHOD TESTS
# ==============================================================================


class TestSample:
    """Test CausalDataset.sample() method."""

    @pytest.fixture()
    def base_data(self):
        """Minimal dataset for sampling tests (no optional fields)."""
        np.random.seed(42)
        n = 200
        return CausalDataset(
            X=np.random.randn(n, 4),
            T=np.random.binomial(1, 0.5, n),
            Y=np.random.randn(n),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
            X_names=["a", "b", "c", "d"],
            T_name="treat",
            Y_name="out",
        )

    @pytest.fixture()
    def full_data(self):
        """Dataset with all optional fields populated."""
        np.random.seed(7)
        n = 200
        return CausalDataset(
            X=np.random.randn(n, 3),
            T=np.random.binomial(1, 0.5, n),
            Y=np.random.randn(n),
            W=np.random.randn(n, 2),
            weights=np.random.uniform(0.5, 1.5, n),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
            X_names=["x1", "x2", "x3"],
            W_names=["w1", "w2"],
            T_name="t",
            Y_name="y",
            true_cates=np.random.randn(n, 1),
        )

    # ------------------------------------------------------------------
    # Return type and size
    # ------------------------------------------------------------------

    def test_sample_returns_causal_dataset(self, base_data):
        """Test that sample() returns a CausalDataset."""
        idx = np.arange(50)
        result = base_data.sample(idx)
        assert isinstance(result, CausalDataset)

    def test_sample_returns_same_type(self, base_data):
        """Test that sample() uses type(self)() so subclasses are preserved."""
        idx = np.arange(50)
        assert type(base_data.sample(idx)) is type(base_data)

    def test_sample_correct_number_of_rows(self, base_data):
        """Test that the sampled dataset has exactly len(indices) rows."""
        idx = np.arange(80)
        result = base_data.sample(idx)
        assert result.X.shape[0] == 80
        assert result.T.shape[0] == 80
        assert result.Y.shape[0] == 80

    def test_sample_single_index_raises_for_continuous_outcome(self, base_data):
        """Test that sampling a single row from a continuous-outcome dataset raises
        ValueError because the resulting dataset has only 1 unique outcome value,
        which fails the continuous-outcome validation (requires > 2 unique values).
        """
        with pytest.raises(ValueError, match="more than 2 unique values"):
            base_data.sample(np.array([5]))

    # ------------------------------------------------------------------
    # Correct values
    # ------------------------------------------------------------------

    def test_sample_X_values_match_original(self, base_data):
        """Test that X values in sampled dataset match original at those indices."""
        idx = np.array([0, 10, 50, 99])
        result = base_data.sample(idx)
        np.testing.assert_array_equal(result.X, base_data.X[idx])

    def test_sample_T_values_match_original(self, base_data):
        """Test that T values in sampled dataset match original at those indices."""
        idx = np.array([1, 20, 75])
        result = base_data.sample(idx)
        np.testing.assert_array_equal(result.T, base_data.T[idx])

    def test_sample_Y_values_match_original(self, base_data):
        """Test that Y values in sampled dataset match original at those indices."""
        idx = np.array([3, 15, 100])
        result = base_data.sample(idx)
        np.testing.assert_array_equal(result.Y, base_data.Y[idx])

    # ------------------------------------------------------------------
    # Optional fields — present
    # ------------------------------------------------------------------

    def test_sample_W_subsetted_when_present(self, full_data):
        """Test that W is subsetted correctly when provided."""
        idx = np.arange(60)
        result = full_data.sample(idx)
        assert result.W is not None
        assert result.W.shape[0] == 60
        np.testing.assert_array_equal(result.W, full_data.W[idx])

    def test_sample_weights_subsetted_when_present(self, full_data):
        """Test that sample weights are subsetted correctly when provided."""
        idx = np.arange(40)
        result = full_data.sample(idx)
        assert result.weights is not None
        assert result.weights.shape[0] == 40
        np.testing.assert_array_equal(result.weights, full_data.weights[idx])

    def test_sample_true_cates_subsetted_when_present(self, full_data):
        """Test that true_cates are subsetted correctly when provided."""
        idx = np.array([0, 5, 10, 15, 20])
        result = full_data.sample(idx)
        assert result.true_cates is not None
        np.testing.assert_array_equal(result.true_cates, full_data.true_cates[idx])

    # ------------------------------------------------------------------
    # Optional fields — absent (None propagated)
    # ------------------------------------------------------------------

    def test_sample_W_none_stays_none(self, base_data):
        """Test that W=None is propagated as None in sampled dataset."""
        result = base_data.sample(np.arange(50))
        assert result.W is None

    def test_sample_weights_none_stays_none(self, base_data):
        """Test that weights=None is propagated as None in sampled dataset."""
        result = base_data.sample(np.arange(50))
        assert result.weights is None

    def test_sample_true_cates_none_stays_none(self, base_data):
        """Test that true_cates=None is propagated as None in sampled dataset."""
        result = base_data.sample(np.arange(50))
        assert result.true_cates is None

    # ------------------------------------------------------------------
    # Metadata preservation
    # ------------------------------------------------------------------

    def test_sample_preserves_treatment_type(self, base_data):
        """Test that treatment_type is preserved after sampling."""
        result = base_data.sample(np.arange(50))
        assert result.treatment_type == base_data.treatment_type

    def test_sample_preserves_outcome_type(self, base_data):
        """Test that outcome_type is preserved after sampling."""
        result = base_data.sample(np.arange(50))
        assert result.outcome_type == base_data.outcome_type

    def test_sample_preserves_feature_names(self, base_data):
        """Test that X_names, T_name, Y_name are preserved after sampling."""
        result = base_data.sample(np.arange(50))
        assert result.X_names == base_data.X_names
        assert result.T_name == base_data.T_name
        assert result.Y_name == base_data.Y_name

    def test_sample_preserves_confounder_names(self, full_data):
        """Test that W_names is preserved after sampling."""
        result = full_data.sample(np.arange(50))
        assert result.W_names == full_data.W_names

    # ------------------------------------------------------------------
    # Index types and patterns
    # ------------------------------------------------------------------

    def test_sample_with_boolean_mask(self, base_data):
        """Test sampling with a boolean index array."""
        mask = np.zeros(200, dtype=bool)
        mask[:100] = True  # first 100 observations
        result = base_data.sample(mask)
        assert result.X.shape[0] == 100
        np.testing.assert_array_equal(result.X, base_data.X[mask])

    def test_sample_with_shuffled_indices(self, base_data):
        """Test that sample() respects index order (no implicit sorting)."""
        idx = np.array([99, 0, 50])
        result = base_data.sample(idx)
        np.testing.assert_array_equal(result.X[0], base_data.X[99])
        np.testing.assert_array_equal(result.X[1], base_data.X[0])
        np.testing.assert_array_equal(result.X[2], base_data.X[50])

    def test_sample_does_not_mutate_original(self, base_data):
        """Test that sample() does not alter the original dataset."""
        original_X = base_data.X.copy()
        _ = base_data.sample(np.arange(50))
        np.testing.assert_array_equal(base_data.X, original_X)

    def test_sample_train_test_split(self, base_data):
        """Test using sample() to produce non-overlapping train/test splits."""
        n = base_data.X.shape[0]
        np.random.seed(0)
        train_idx = np.random.choice(n, 150, replace=False)
        test_idx = np.setdiff1d(np.arange(n), train_idx)

        train = base_data.sample(train_idx)
        test = base_data.sample(test_idx)

        assert train.X.shape[0] == 150
        assert test.X.shape[0] == n - 150
        # No overlap in rows
        assert train.X.shape[0] + test.X.shape[0] == n
