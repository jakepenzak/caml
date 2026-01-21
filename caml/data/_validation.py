from __future__ import annotations

import numpy as np
import pandas as pd

from caml.data.schema import OutcomeType, TreatmentType


def check_shapes_match(X, T, Y, W=None) -> None:
    """Verify all arrays have compatible shapes."""
    n_obs = _n_obs(T)
    _check_length("Y", Y, n_obs)
    _check_length("X", X, n_obs)
    if W is not None:
        _check_length("W", W, n_obs)


def check_1d_targets(T, Y) -> None:
    """Ensure treatment and outcome are 1-dimensional."""
    _check_1d_like("T", T)
    _check_1d_like("Y", Y)


def check_missing_data(X, T, Y, W=None) -> dict:
    """Check for missing values and return summary."""
    summary = {
        "X": _count_missing(X),
        "T": _count_missing(T),
        "Y": _count_missing(Y),
        "W": _count_missing(W) if W is not None else 0,
    }
    return summary


def check_treatment_type_matches_data(T, declared_type: TreatmentType) -> None:
    """Verify the declared treatment type matches actual data."""
    values = _unique_non_null(T)
    if declared_type == TreatmentType.BINARY:
        if len(values) > 2:
            raise ValueError("TreatmentType.BINARY expects at most 2 unique values.")
    elif declared_type == TreatmentType.MULTI:
        if len(values) <= 2:
            raise ValueError("TreatmentType.MULTI expects more than 2 unique values.")
    elif declared_type == TreatmentType.CONTINUOUS:
        if not _is_numeric(T):
            raise ValueError(
                "TreatmentType.CONTINUOUS expects numeric treatment values."
            )


def check_outcome_type_matches_data(Y, declared_type: OutcomeType) -> None:
    values = _unique_non_null(Y)
    if declared_type == OutcomeType.BINARY:
        if len(values) > 2:
            raise ValueError("OutcomeType.BINARY expects at most 2 unique values.")
    elif declared_type == OutcomeType.CONTINUOUS:
        if not _is_numeric(Y):
            raise ValueError(
                "TreatmentType.CONTINUOUS expects numeric treatment values."
            )


def _n_obs(arr) -> int:
    if arr is None:
        raise ValueError("Expected array-like input, got None.")
    return len(arr)


def _check_length(name: str, arr, n_obs: int) -> None:
    if arr is None:
        return
    if len(arr) != n_obs:
        raise ValueError(f"{name} must have length {n_obs}, got {len(arr)}.")


def _check_1d_like(name: str, arr) -> None:
    if isinstance(arr, pd.Series):
        return
    if isinstance(arr, pd.DataFrame):
        if arr.shape[1] == 1:
            return
        raise ValueError(f"{name} must be 1-dimensional, got {arr.shape[1]} columns.")
    if isinstance(arr, np.ndarray):
        if arr.ndim == 1:
            return
        if arr.ndim == 2 and arr.shape[1] == 1:
            return
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}.")
    raise TypeError(f"{name} must be a pandas or numpy array-like.")


def _count_missing(arr) -> int:
    if arr is None:
        return 0
    missing = pd.isna(arr)
    if isinstance(missing, (pd.Series, pd.DataFrame)):
        return int(missing.to_numpy().sum())
    return int(np.sum(missing))


def _unique_non_null(arr) -> np.ndarray:
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        values = arr.to_numpy().ravel()
    else:
        values = np.asarray(arr).ravel()
    values = values[~pd.isna(values)]
    if values.size == 0:
        return values
    return np.unique(values)


def _is_numeric(arr) -> bool:
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        return pd.api.types.is_numeric_dtype(arr.dtypes)
    return np.issubdtype(np.asarray(arr).dtype, np.number)
