from __future__ import annotations

from typing import Literal

Estimand = Literal[
    "ate",
    "att",
    "atc",
    "cate",
    "gate",
    "gatt",
    "event_study",
]

_SUPPORTED = set(Estimand.__args__)  # type: ignore[attr-defined]


def normalize(estimand: str) -> Estimand:
    e = estimand.strip().lower()
    if e not in _SUPPORTED:
        raise ValueError(
            f"Unknown estimand: {estimand}. Supported: {sorted(_SUPPORTED)}"
        )
    return e  # type: ignore[return-value]
