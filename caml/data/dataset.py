"""Unified causal data container for CATE estimation.

Provides ``CausalDataset``, the core data structure for causal inference workflows in CaML.
Encapsulates treatment variables, outcomes, effect modifiers, and confounders with automatic
validation and metadata tracking.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from caml.data._validation import (
    check_1d_targets,
    check_missing_data,
    check_outcome_type_matches_data,
    check_shapes_match,
    check_treatment_type_matches_data,
)
from caml.data.data_schema import OutcomeType, TreatmentType


@dataclass
class CausalDataset:
    """Unified causal data container with validation and metadata.

    Encapsulates all data required for CATE estimation: treatment, outcome, effect
    modifiers, and optional confounders. Performs automatic validation on initialization
    and tracks metadata for estimator compatibility checking.

    Parameters
    ----------
    X
        Effect modifiers for heterogeneity estimation (features for CATE).
    T
        Treatment variable (must be 1-dimensional).
    Y
        Outcome variable (must be 1-dimensional).
    W
        Additional confounders not used for heterogeneity modeling.
    weights
        Sample weights for weighted estimation.
    treatment_type
        Treatment type (``TreatmentType.BINARY``, ``MULTI``, or ``CONTINUOUS``).
    outcome_type
        Outcome type (``OutcomeType.BINARY`` or ``CONTINUOUS``).
    X_names
        Effect modifier names (auto-set by ``from_dataframe()``).
    W_names
        Confounder names (auto-set by ``from_dataframe()``).
    T_name
        Treatment name.
    Y_name
        Outcome name.

    Raises
    ------
    ValueError
        If validation fails (shape mismatch, wrong types, missing values).

    See Also
    --------
    [`TreatmentType`](data_schema.qmd#caml.data.data_schema.TreatmentType) : Treatment variable categories.

    [`OutcomeType`](data_schema.qmd#caml.data.data_schema.OutcomeType) : Outcome variable categories.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol for CATE estimators.

    Notes
    -----
    - Validation is automatic via ``__post_init__``
    - Supports both pandas and numpy data structures
    - T and Y must be 1-dimensional

    Examples
    --------
    Create from arrays:

    ```{python}
    import numpy as np
    from caml.data import CausalDataset, TreatmentType, OutcomeType

    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 3)
    T = np.random.binomial(1, 0.5, n)
    Y = X[:, 0] + 0.5 * T + np.random.randn(n)

    data = CausalDataset(
        X=X, T=T, Y=Y,
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )
    print(f"Dataset: {len(data.Y)} observations")
    ```

    Create from DataFrame (recommended):

    ```{python}
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(n_cont_modifiers=3, seed=42)
    df = gen.df

    data = CausalDataset.from_dataframe(
        df,
        X=[c for c in df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )
    print(f"Effect modifiers: {data.X_names}")
    ```
    """

    # Core data (always required)
    X: pd.Series | pd.DataFrame | np.ndarray  # Effect Modifiers and/or confounders
    T: pd.Series | pd.DataFrame | np.ndarray  # Treatment
    Y: pd.Series | pd.DataFrame | np.ndarray  # Outcome

    # Optional Data
    W: pd.Series | pd.DataFrame | np.ndarray | None = None
    weights: np.ndarray | None = None

    # Metadata
    treatment_type: TreatmentType = field(default=TreatmentType.BINARY)
    outcome_type: OutcomeType = field(default=OutcomeType.CONTINUOUS)

    # Feature Names
    X_names: list[str] | None = None
    W_names: list[str] | None = None
    T_name: str = "treatment"
    Y_name: str = "outcome"

    def validate(self) -> None:
        """Perform validation checks on the dataset.

        Validates shapes, dimensionality, missing values, and treatment/outcome types.

        Raises
        ------
        ValueError
            If any validation check fails.

        Notes
        -----
        Automatically called during ``__post_init__``. Manual calls typically unnecessary.

        Examples
        --------
        ```{python}
        import numpy as np
        from caml.data import CausalDataset, TreatmentType

        # This raises ValueError - wrong treatment type
        try:
            data = CausalDataset(
                X=np.random.randn(100, 3),
                T=np.random.choice([0, 1, 2], 100),  # 3 values
                Y=np.random.randn(100),
                treatment_type=TreatmentType.BINARY  # Declared as binary!
            )
        except ValueError as e:
            print(f"Validation error: {e}")
        ```
        """
        check_shapes_match(self.X, self.T, self.Y, self.W)
        check_1d_targets(self.T, self.Y)
        check_missing_data(self.X, self.T, self.Y, self.W)
        check_treatment_type_matches_data(self.T, self.treatment_type)
        check_outcome_type_matches_data(self.Y, self.outcome_type)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        X: list[str],
        T: str,
        Y: str,
        W: list[str] | None = None,
        treatment_type: TreatmentType = TreatmentType.BINARY,
        outcome_type: OutcomeType = OutcomeType.CONTINUOUS,
        **kwargs,
    ) -> "CausalDataset":
        """Construct ``CausalDataset`` from a pandas DataFrame.

        Recommended way to create a ``CausalDataset``. Automatically extracts and tracks
        column names for interpretability.

        Parameters
        ----------
        df
            Source DataFrame containing all variables.
        X
            Column names for effect modifiers (features for CATE).
        T
            Column name for treatment variable.
        Y
            Column name for outcome variable.
        W
            Column names for confounders not used in heterogeneity modeling.
        treatment_type
            Treatment type.
        outcome_type
            Outcome type.
        **kwargs
            Additional arguments (e.g., ``weights``).

        Returns
        -------
        CausalDataset
            Initialized and validated dataset.

        Raises
        ------
        KeyError
            If column names not found in ``df``.
        ValueError
            If validation fails.

        Notes
        -----
        Column names are automatically stored in ``X_names``, ``W_names``, ``T_name``, ``Y_name``.

        Examples
        --------
        ```{python}
        import pandas as pd
        from caml.data import CausalDataset, TreatmentType, OutcomeType

        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'treated': [0, 1, 0, 1, 1],
            'outcome': [100, 150, 120, 180, 200]
        })

        data = CausalDataset.from_dataframe(
            df,
            X=['age', 'income'],
            T='treated',
            Y='outcome',
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS
        )
        print(f"Effect modifiers: {data.X_names}")
        ```
        """
        return cls(
            X=df[X],
            T=df[T],
            Y=df[Y],
            W=df[W] if W else None,
            treatment_type=treatment_type,
            outcome_type=outcome_type,
            X_names=X,
            W_names=W,
            T_name=T,
            Y_name=Y,
            **kwargs,
        )

    def __post_init__(self):
        """Run validations after dataclass initialization.

        Automatically called to ensure data is validated immediately upon construction.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        self.validate()
