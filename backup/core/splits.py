from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


@dataclass(frozen=True)
class SplitSpec:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = None
    group_col: str | None = None


def iter_splits(
    df: pd.DataFrame, spec: SplitSpec
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(df))
    if spec.group_col:
        groups = df[spec.group_col].to_numpy()
        splitter = GroupKFold(n_splits=spec.n_splits)
        yield from splitter.split(idx, groups=groups)
        return

    splitter = KFold(
        n_splits=spec.n_splits, shuffle=spec.shuffle, random_state=spec.random_state
    )
    yield from splitter.split(idx)
