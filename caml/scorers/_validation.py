import numpy as np


def _clip(arr: np.ndarray, lb: float = 0.01, ub: float = np.inf) -> np.ndarray:
    """Clip array values (commonly propensity scores) for stability.

    Used to trim propensity scores when they appear in denominators (e.g., IPW,
    DR), preventing extreme weights.

    Parameters
    ----------
    arr
        Array to clip
    lb
        Lower bound
    ub
        Upper bound

    Returns
    -------
    np.ndarray
        Clipped array
    """
    return np.clip(arr, lb, ub)


def _validate_cate_array(
    arr: np.ndarray,
    n_samples: int,
    name: str = "CATE predictions",
) -> np.ndarray:
    """Validate and flatten CATE array to 1D.

    Ensures CATE predictions have the correct number of samples and converts
    to 1D array for consistent downstream computation. Handles common shape
    variations from different estimators (e.g., ``(n,)``, ``(n, 1)``).

    Parameters
    ----------
    arr
        Array of CATE predictions to validate.
    n_samples
        Expected number of samples.
    name
        Name of the array for error messages (e.g., "tau_hat", "true_cates").

    Returns
    -------
    np.ndarray
        1D array of shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If array has wrong number of samples or incompatible shape.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.scorers import validate_cate_array

    # 2D array with shape (100, 1) -> flattened to (100,)
    arr_2d = np.random.randn(100, 1)
    arr_1d = validate_cate_array(arr_2d, n_samples=100, name="tau_hat")
    print(arr_1d.shape)  # (100,)

    # Already 1D array passes through
    arr = np.random.randn(100)
    result = validate_cate_array(arr, n_samples=100)
    print(result.shape)  # (100,)
    ```
    """
    arr = np.asarray(arr)

    # Handle 2D arrays with single column (common from estimators)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr.ravel()
        else:
            raise ValueError(
                f"{name} has invalid shape {arr.shape}. Expected 1D array or 2D with "
                f"single column (n, 1), but got {arr.shape[1]} columns."
            )

    # Validate 1D shape
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1D or 2D with single column, got {arr.ndim}D array "
            f"with shape {arr.shape}."
        )

    # Validate number of samples
    if arr.shape[0] != n_samples:
        raise ValueError(f"{name} has {arr.shape[0]} samples, expected {n_samples}.")

    return arr


def _validate_scorer_inputs(
    tau_hat: np.ndarray,
    reference: np.ndarray,
    tau_name: str = "CATE predictions",
    ref_name: str = "reference",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and align shapes of CATE predictions and reference array.

    Ensures both arrays have the same number of samples and converts to 1D
    for consistent computation. This is the primary validation function for
    scorer ``__call__`` methods.

    Parameters
    ----------
    tau_hat
        Estimated CATE values from estimator.
    reference
        Reference array to compare against (e.g., true_cates, pseudo-outcome).
    tau_name
        Name for tau_hat in error messages.
    ref_name
        Name for reference in error messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (tau_hat, reference) as validated 1D arrays.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes or different sample counts.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.scorers import validate_scorer_inputs

    # Different shapes but same n_samples -> both flattened
    tau = np.random.randn(100, 1)
    ref = np.random.randn(100)
    tau_flat, ref_flat = validate_scorer_inputs(tau, ref)
    print(tau_flat.shape, ref_flat.shape)  # (100,) (100,)
    ```
    """
    reference = np.asarray(reference)

    # Determine expected n_samples from reference
    if reference.ndim == 1:
        n_samples = reference.shape[0]
    elif reference.ndim == 2 and reference.shape[1] == 1:
        n_samples = reference.shape[0]
    else:
        raise ValueError(
            f"{ref_name} has invalid shape {reference.shape}. Expected 1D array "
            f"or 2D with single column."
        )

    # Validate and flatten both arrays
    tau_hat = _validate_cate_array(tau_hat, n_samples, tau_name)
    reference = _validate_cate_array(reference, n_samples, ref_name)

    return tau_hat, reference
