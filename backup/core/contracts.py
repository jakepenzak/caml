from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd

Estimand = Literal[
    "ate",
    "att",
    "atc",
    "cate",
    "gate",
    "gatt",
    "event_study",
]


@dataclass(frozen=True)
class InferenceSpec:
    cov_type: str = "nonrobust"
    cluster: str | None = None
    level: float = 0.95
    method: str = "wald"


@dataclass
class EffectResult:
    estimand: Estimand
    value: Any
    stderr: Any | None = None
    inference: InferenceSpec = field(default_factory=InferenceSpec)
    n_obs: int | None = None
    n_clusters: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    spec_: dict[str, Any] = field(default_factory=dict)
    backend_: Any | None = None

    def ci(self, level: float | None = None) -> tuple[Any, Any]:
        lvl = level if level is not None else self.inference.level
        if self.stderr is None:
            raise ValueError("No stderr available for CI")

        # MVP: normal approximation
        z = 1.96 if abs(lvl - 0.95) < 1e-12 else 1.96
        return (self.value - z * self.stderr, self.value + z * self.stderr)


class CausalEstimator(Protocol):
    def fit(self, df: pd.DataFrame, **kwargs) -> "CausalEstimator": ...

    def estimate(
        self, estimand: Estimand, df: pd.DataFrame | None = None, **kwargs
    ) -> EffectResult: ...

    def effect(self, df: pd.DataFrame | None = None, **kwargs) -> Any: ...

    def get_diagnostics(self) -> dict[str, Any]: ...
