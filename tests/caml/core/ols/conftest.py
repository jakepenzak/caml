import numpy as np
import pandas as pd
import pytest

import caml.core.ols as ols_mod

pytestmark = [pytest.mark.core, pytest.mark.ols]


@pytest.fixture(params=["jax", "numpy"], ids=["with_jax", "with_numpy"])
def backend(request, monkeypatch):
    """Simulate two backends.

    - 'jax'   → real jax, jax.numpy, jax.scipy.stats, _HAS_JAX=True
    - 'numpy' → numpy, scipy.stats,    _HAS_JAX=False.
    """
    if request.param == "jax":
        import jax
        import jax.numpy as jnp
        import jax.scipy.stats as jstats

        monkeypatch.setattr(ols_mod, "_HAS_JAX", True)
        monkeypatch.setattr(ols_mod, "jax", jax)
        monkeypatch.setattr(ols_mod, "jnp", jnp)
        monkeypatch.setattr(ols_mod, "jstats", jstats)
    else:
        import numpy as np
        import scipy.stats as sstats

        monkeypatch.setattr(ols_mod, "_HAS_JAX", False)
        monkeypatch.setattr(ols_mod, "jax", None)
        monkeypatch.setattr(ols_mod, "jnp", np)
        monkeypatch.setattr(ols_mod, "jstats", sstats)

    return request.param


@pytest.fixture
def multi_df():
    """Generate multi-dimensional outcome dataframe.

    A DataFrame with two continuous outcomes (Y1,Y2),
    one binary treatment T, two heterogeneity covariates X1,X2,
    and one control W.
    """
    rng = np.random.default_rng(123)
    n = 200

    # binary treatment
    T = rng.integers(0, 2, size=n)
    # covariates
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n) * 2
    W = rng.normal(size=n)

    # outcomes with known linear relationships
    Y1 = 1.5 * T + 0.5 * X1 - 0.2 * X2 + 0.1 * W + rng.normal(scale=0.1, size=n)
    Y2 = -0.7 * T + 0.3 * X1 + 0.8 * X2 + 0.2 * W + rng.normal(scale=0.1, size=n)

    return pd.DataFrame(
        {
            "T": T,
            "X1": X1,
            "X2": X2,
            "W": W,
            "Y1": Y1,
            "Y2": Y2,
        }
    )


@pytest.fixture
def DummyJax():
    class DummyJax:
        @staticmethod
        def devices(kind):
            # Simulate JAX running, but no GPU devices available:
            raise RuntimeError("No GPU found")

    return DummyJax()
