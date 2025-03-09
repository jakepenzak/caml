from functools import wraps
from typing import Iterable

import pandas as pd
import patsy
from joblib import Parallel, delayed
from typeguard import typechecked

from ..generics import experimental, timer
from ..logging import DEBUG, ERROR, INFO, WARNING

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.stats as jstats

    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    import numpy as jnp
    import scipy.stats as jstats

    _HAS_JAX = False


@experimental
@typechecked
class FastLeastSquares:
    r"""FastLeastSquares is a fast implementation of the Least Squares estimator designed specifically with Causal Inference in mind.

    This estimator estimates a standard linear regression model for any number of continuous or binary outcomes and a single binary treatment,
    and provides estimates for the Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs) out of the box. Additionally,
    methods are provided for estimating a specific Conditional Average Treatment Effect (CATE), as well as individual Conditional Average Treatment
    Effects (CATEs) for a group of observations.

    **Outcome/Treatment Type Support Matrix**
    <center>
    | Outcome     | Treatment   | Support     | Missing    |
    | ----------- | ----------- | ----------- | ---------- |
    | Continuous  | Binary      | ✅Full      |            |
    | Continuous  | Continuous  | ❌Not yet   |            |
    | Continuous  | Categorical | ❌Not yet   |            |
    | Binary      | Binary      | ✅Full      |            |
    | Binary      | Continuous  | ❌Not yet   |            |
    | Binary      | Categorical | ❌Not yet   |            |
    | Categorical | Binary      | ❌Not yet   |            |
    | Categorical | Continuous  | ❌Not yet   |            |
    | Categorical | Categorical | ❌Not yet   |            |
    </center>

    Parameters
    ----------
    Y
        A list of outcome variable names.
    T
        The treatment variable name.
    G
        A list of group variable names, by default None. These will be the groups for which GATEs will be estimated.
    X
        A list of covariate variable names, by default None. These will be the covariates for which heterogeneity/CATEs can be estimated.
    W
        A list of instrument variable names, by default None. These will be the additional covariates not used for modeling heterogeneity/CATEs.
    discrete_treatment
        Whether the treatment is discrete, by default True
    engine
        The engine to use for computation, by default "cpu". Can be "cpu" or "gpu". Note "gpu" requires JAX to be installed, which can be installed
        via `pip install caml[jax-gpu]`.

    Attributes
    ----------
    Y
        A list of outcome variable names.
    T
        The treatment variable name.
    G
        A list of group variable names, by default None. These will be the groups for which GATEs will be estimated.
    X
        A list of covariate variable names, by default None. These will be the covariates for which heterogeneity/CATEs can be estimated.
    W
        A list of instrument variable names, by default None. These will be the additional covariates not used for modeling heterogeneity/CATEs.
    discrete_treatment
        Whether the treatment is discrete, by default True
    engine
        The engine to use for computation, by default "cpu". Can be "cpu" or "gpu". Note "gpu" requires JAX to be installed, which can be installed
        via `pip install caml[jax-gpu]`
    formula
        The formula leveraged for design matrix creation via Patsy.
    results
        A dictionary containing the results of the fitted model & estimated ATEs/GATEs.
    """

    def __init__(
        self,
        Y: Iterable[str],
        T: str,
        G: Iterable[str] | None = None,
        X: Iterable[str] | None = None,
        W: Iterable[str] | None = None,
        *,
        discrete_treatment: bool = True,
        engine: str = "cpu",
    ):
        DEBUG(
            f"Initializing {self.__class__.__name__} with parameters: Y={Y}, T={T}, G={G}, X={X}, W={W}, discrete_treatment={discrete_treatment}, engine={engine}"
        )
        self.Y = Y
        self.T = T
        self.G = G
        self.X = X
        self.W = W
        self.discrete_treatment = discrete_treatment

        if engine not in ["cpu", "gpu"]:
            ERROR(
                f"Invalid engine specified: {engine}. Only 'cpu' and 'gpu' are supported."
            )
            raise ValueError("Only 'cpu' and 'gpu' are supported for engine argument")

        if engine == "gpu":
            if not _HAS_JAX:
                ERROR("GPU engine requested but JAX is not available")
                raise ValueError("JAX is required for gpu engine.")
            try:
                len(jax.devices("gpu"))
            except RuntimeError:
                WARNING("No available GPU detected, falling back to CPU")
                engine = "cpu"

        self.engine = engine
        self.formula = self._create_formula(self.Y, self.T, self.G, self.X, self.W)
        DEBUG(f"Created formula: {self.formula}")
        self._fitted = False
        self.results = {}

    def fit(self, data, n_jobs: int | None = None):
        pd_df = self.convert_dataframe_to_pandas(data, self.G)
        y, X = self._create_design_matrix(pd_df)
        self._fit(X, y)
        diff_matrix = self._create_design_matrix(pd_df, create_diff_matrix=True)
        self._estimate_ates(pd_df, diff_matrix)
        self._estimate_gates(pd_df, diff_matrix, n_jobs)
        self._fitted = True

    @timer("Single CATE Estimation")
    def estimate_single_cate(self, data) -> dict:
        if not self._fitted:
            raise RuntimeError("You must call .fit() prior to calling this method.")

        data = self.convert_dataframe_to_pandas(data, self.G)
        diff_matrix = self._create_design_matrix(data, create_diff_matrix=True)

        INFO("Estimating single Conditional Average Treatment Effect (CATE)...")

        n_treated = int(data[self.T].sum())

        statistics = self._compute_statistics(
            diff_matrix=diff_matrix,
            params=self.results["params"],
            vcv=self.results["vcv"],
            n_treated=n_treated,
        )

        results = {"outcomes": self.Y}
        results.update({key: statistics[key] for key in statistics.keys()})

        return results

    def estimate_cates(self, data) -> dict:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support estimate_cates method yet."
        )

    @timer("Model Fitting")
    def _fit(self, X: jnp.ndarray, y: jnp.ndarray):
        INFO("Fitting regression model...")

        @maybe_jit
        def fit(X, y):
            params, _, _, _ = jnp.linalg.lstsq(X, y, rcond=None)
            y_resid = y - X @ params
            rss = jnp.sum(y_resid**2, axis=0)
            sigma_squared_hat = rss / (X.shape[0] - X.shape[1])
            XtX_inv = jnp.linalg.pinv(X.T @ X)
            vcv = (
                sigma_squared_hat[:, jnp.newaxis, jnp.newaxis]
                * XtX_inv[jnp.newaxis, :, :]
            )
            return params, vcv

        params, vcv = fit(X, y)

        self.results["params"] = params
        self.results["vcv"] = vcv
        self.results["std_err"] = jnp.sqrt(jnp.diagonal(vcv, axis1=1, axis2=2))
        self.results["treatment_effects"] = {}

    @timer("ATE Estimation")
    def _estimate_ates(self, data, diff_matrix: jnp.ndarray | None = None):
        INFO("Estimating Average Treatment Effects (ATEs)...")

        treated_mask = jnp.array(data[self.T] == 1)
        n_treated = int(treated_mask.sum())

        statistics = self._compute_statistics(
            diff_matrix=diff_matrix,
            params=self.results["params"],
            vcv=self.results["vcv"],
            n_treated=n_treated,
        )

        self.results["treatment_effects"]["overall"] = {"outcomes": self.Y}
        self.results["treatment_effects"]["overall"].update(
            {key: statistics[key] for key in statistics.keys()}
        )

    @timer("GATE Estimation")
    def _estimate_gates(
        self,
        data,
        diff_matrix: jnp.ndarray | None = None,
        n_jobs: int | None = None,
    ):
        if self.G is None:
            DEBUG("No groups specified for GATE estimation. Skipping.")
            return

        INFO("Estimating Group Average Treatment Effects (GATEs)...")

        groups = {group: data[group].unique() for group in self.G}

        # Prepare groups for processing
        group_info = []
        for group in groups:
            for membership in groups[group]:
                mask = jnp.array(data[group] == membership)
                treated_mask = jnp.array(data[data[group] == membership][self.T] == 1)
                group_key = f"{group}-{membership}"
                group_info.append((group_key, group, membership, mask, treated_mask))

        params = self.results["params"]
        vcv = self.results["vcv"]

        def process_group(group_key, mask, treated_mask):
            diff_matrix_filtered = diff_matrix[mask]
            n_treated = int(treated_mask.sum())
            statistics = self._compute_statistics(
                diff_matrix=diff_matrix_filtered,
                params=params,
                vcv=vcv,
                n_treated=n_treated,
            )
            return group_key, statistics

        DEBUG(f"Starting parallel processing with {n_jobs} jobs")
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_group)(group_key, mask, treated_mask)
            for group_key, _, _, mask, treated_mask in group_info
        )

        for group_key, statistics in results:
            self.results["treatment_effects"][group_key] = {"outcomes": self.Y}
            self.results["treatment_effects"][group_key].update(
                {key: statistics[key] for key in statistics.keys()}
            )

    @timer("Design Matrix Creation")
    def _create_design_matrix(
        self, data: pd.DataFrame, create_diff_matrix: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        if create_diff_matrix:
            DEBUG("Creating treatment difference matrix...")
            original_t = data[self.T].copy()
            data[self.T] = 1
            X1 = patsy.dmatrix(self._X_design_info, data=data)
            data[self.T] = 0
            X0 = patsy.dmatrix(self._X_design_info, data=data)
            data[self.T] = original_t

            if _HAS_JAX:
                X1 = jnp.array(X1, device=jax.devices(self.engine)[0])
                X0 = jnp.array(X0, device=jax.devices(self.engine)[0])
            else:
                X1 = jnp.array(X1)
                X0 = jnp.array(X0)

            diff = X1 - X0

            return diff
        else:
            DEBUG("Creating model matrix...")
            y, X = patsy.dmatrices(self.formula, data=data)

            self._X_design_info = X.design_info

            if _HAS_JAX:
                y = jnp.array(y, device=jax.devices(self.engine)[0])
                X = jnp.array(X, device=jax.devices(self.engine)[0])
            else:
                y = jnp.array(y)
                X = jnp.array(X)

            return y, X

    @staticmethod
    def convert_dataframe_to_pandas(dataframe, groups) -> pd.DataFrame:
        def convert_groups_to_categorical(df, groups):
            for col in groups or []:
                df[col] = df[col].astype("category")
            return df

        if isinstance(dataframe, pd.DataFrame):
            return convert_groups_to_categorical(dataframe, groups)

        DEBUG(f"Converting input dataframe of type {type(dataframe)} to pandas")

        if hasattr(dataframe, "toPandas"):
            return convert_groups_to_categorical(dataframe.toPandas(), groups)

        if hasattr(dataframe, "to_pandas"):
            return convert_groups_to_categorical(dataframe.to_pandas(), groups)

        ERROR(f"Unsupported dataframe type: {type(dataframe)}")
        raise Exception(
            f"Pandas conversion not currently supported for {type(dataframe)}."
        )

    @staticmethod
    def _compute_statistics(
        diff_matrix: jnp.ndarray,
        params: jnp.ndarray,
        vcv: jnp.ndarray,
        n_treated: int,
    ) -> dict:
        diff_mean = jnp.mean(diff_matrix, axis=0)

        ate = diff_mean @ params
        std_err = jnp.sqrt(diff_mean @ vcv @ diff_mean.T)
        t_stat = jnp.where(std_err > 0, ate / std_err, 0)
        pval = 2 * (1 - jstats.norm.cdf(t_stat))
        n = diff_matrix.shape[0]
        n_control = n - n_treated

        return {
            "ate": ate,
            "std_err": std_err,
            "t_stat": t_stat,
            "pval": pval,
            "n": n,
            "n_treated": n_treated,
            "n_control": n_control,
        }

    @staticmethod
    def _create_formula(
        Y: list[str],
        T: str,
        G: list[str] | None,
        X: list[str] | None = None,
        W: list[str] | None = None,
    ) -> str:
        formula = " + ".join(Y)
        formula += f" ~ {T}"

        for g in G or []:
            formula += f" + C({g})*{T}"

        for x in X or []:
            formula += f" + {x}*{T}"

        for w in W or []:
            formula += f" + {w}"

        return formula


def maybe_jit(func=None, **jit_kwargs):
    def maybe_jit_inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _HAS_JAX:
                return jax.jit(func, **jit_kwargs)(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    if func is None:
        return maybe_jit_inner

    return maybe_jit_inner(func)
