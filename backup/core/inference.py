from __future__ import annotations

from typing import Any

import pandas as pd


def cluster_info(
    df: pd.DataFrame, cluster_col: str | None
) -> tuple[int | None, int | None]:
    if cluster_col is None:
        return (len(df), None)
    return (len(df), int(df[cluster_col].nunique(dropna=True)))


def wald_ci(value: Any, stderr: Any, level: float = 0.95) -> tuple[Any, Any]:
    z = 1.96
    return (value - z * stderr, value + z * stderr)
