from enum import Enum


class TreatmentType(Enum):
    """Categories of treatment variables."""

    BINARY = "binary"
    MULTI = "multi"
    CONTINUOUS = "continuous"

    def is_discrete(self) -> bool:
        return self in {TreatmentType.BINARY, TreatmentType.MULTI}


class OutcomeType(Enum):
    """Categories of outcome variables."""

    BINARY = "binary"
    CONTINUOUS = "continuous"

    def is_discrete(self) -> bool:
        return self == OutcomeType.BINARY


class Estimand(Enum):
    """Categories of target estimands."""

    ATE = "ate"  # Average Treatment Effect
    ATT = "att"  # Average Treatment Effect on Treated
    ATC = "atc"  # Average Treatment Effect on Control
    CATE = "cate"  # Conditional ATE
    GATE = "gate"  # Group ATE
