"""Tests for caml.data.dataset module."""

import numpy as np
import pandas as pd
import pytest

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.extensions.synthetic_data import SyntheticDataGenerator

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
                "x1": [1, 2],
                "w1": [3, 4],
                "w2": [5, 6],
                "t": [0, 1],
                "y": [10, 20],
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

        data = CausalDataset(X=X, T=T, Y=Y)

        assert len(data.Y) == 1

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
