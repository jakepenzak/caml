from enum import Enum


class InferenceType(Enum):
    """Categories of Supported Inference Types."""

    ANALYTIC = "analytic"
    BOOTSTRAP = "bootstrap"
