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
from caml.data.schema import OutcomeType, TreatmentType


@dataclass
class CausalDataset:
    """Unified causal data container."""

    # Core data (always required)
    X: pd.Series | pd.DataFrame | np.ndarray  # Effect Modifiers and/or confounders
    T: pd.Series | pd.DataFrame | np.ndarray  # Treatment
    Y: pd.Series | pd.DataFrame | np.ndarray  # Outcome

    # Optional Data
    W: pd.Series | pd.DataFrame | np.ndarray | None = None
    weights: np.ndarray | None = None

    # Grouping/clustering
    cluster_id: np.ndarray | None = None

    # Metadata
    treatment_type: TreatmentType = field(default=TreatmentType.BINARY)
    outcome_type: OutcomeType = field(default=OutcomeType.CONTINUOUS)

    # Feature Names
    X_names: list[str] | None = None
    W_names: list[str] | None = None
    T_name: str = "treatment"
    Y_name: str = "outcome"

    def validate(self) -> None:
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
        """Run validations post-init."""
        self.validate()
