from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

DataStructure = Literal["iid"]
Design = Literal["unconfoundedness"]


@dataclass(frozen=True)
class CausalDataSpec:
    outcome: str
    treatment: str
    covariates: Sequence[str] = ()
    heterogeneity_features: Sequence[str] = ()
    weights: str | None = None

    # unit: str | None = None
    # time: str | None = None
    # cluster: str | None = None


@dataclass(frozen=True)
class DesignSpec:
    design: Design
    data_structure: DataStructure

    # # DiD
    # event_window: tuple[int, int] = (-5, 5)
    # anticipation: int = 0

    # add_unit_fe: bool = True
    # add_time_fe: bool = True


def resolve_spec(data: CausalDataSpec, design: DesignSpec) -> dict:
    hetero = list(data.heterogeneity_features) or list(data.covariates)

    return {
        "design": design.design,
        "data_structure": design.data_structure,
        "outcome": data.outcome,
        "treatment": data.treatment,
        "covariates": list(data.covariates),
        "heterogeneity_features": hetero,
        "weights": data.weights,
        # "unit": data.unit,
        # "time": data.time,
        # "cluster": cluster,
        # "event_window": design.event_window,
        # "anticipation": design.anticipation,
        # "add_unit_fe": design.add_unit_fe,
        # "add_time_fe": design.add_time_fe,
    }
