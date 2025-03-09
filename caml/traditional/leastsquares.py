import timeit
from functools import wraps
from typing import Iterable

import pandas as pd
import patsy
import psutil
from joblib import Parallel, cpu_count, delayed

from ..generics import cls_typechecked, experimental
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
@cls_typechecked
class FastLeastSquares:
    """FastLeastSquares is a fast implementation of the LSTSQ estimator."""

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

    def fit(self, data, n_jobs: int | None = None, memory_limit: float = 0.8):
        pd_df = self.convert_dataframe_to_pandas(data, self.G)
        self._fit(pd_df)
        diff_matrix = self._create_diff_matrix(pd_df)
        self._estimate_ates(pd_df, diff_matrix)
        self._estimate_gates(pd_df, diff_matrix, n_jobs, memory_limit)

    def estimate_single_cate(self, data) -> dict:
        if not self._fitted:
            raise RuntimeError("You must call .fit() prior to calling this method.")

        data = self.convert_dataframe_to_pandas(data, self.G)
        diff_matrix = self._create_diff_matrix(data)

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

    def _fit(self, data):
        data = self.convert_dataframe_to_pandas(data, self.G)

        start_time = timeit.default_timer()
        INFO("Converting data to design matrix...")
        y, X = patsy.dmatrices(self.formula, data=data)
        self._X_design_info = X.design_info
        end_time = timeit.default_timer()
        DEBUG(
            f"Design matrix conversion completed in {end_time - start_time:.2f} seconds"
        )

        DEBUG(f"Moving data to {self.engine}")
        start_time = timeit.default_timer()
        if _HAS_JAX:
            y = jnp.array(y, device=jax.devices(self.engine)[0])
            X = jnp.array(X, device=jax.devices(self.engine)[0])
            DEBUG(f"Data shape - X: {X.shape}, y: {y.shape}")
        else:
            y = jnp.array(y)
            X = jnp.array(X)
        end_time = timeit.default_timer()
        DEBUG(f"Data transfer completed in {end_time - start_time:.2f} seconds")

        INFO("Fitting regression model...")
        start_time = timeit.default_timer()

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

        self._fitted = True
        end_time = timeit.default_timer()
        DEBUG(f"Model fitting completed in {end_time - start_time:.2f} seconds")

    def _estimate_ates(self, data, diff_matrix: jnp.ndarray | None = None):
        if not self._fitted:
            ERROR("Attempting to estimate ATEs before model is fitted")
            raise RuntimeError("You must call .fit() prior to calling this method.")

        data = self.convert_dataframe_to_pandas(data, self.G)

        if diff_matrix is None:
            diff_matrix = self._create_diff_matrix(data)

        INFO("Estimating Average Treatment Effects (ATEs)...")
        start_time = timeit.default_timer()

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

        end_time = timeit.default_timer()
        DEBUG(f"ATE estimation completed in {end_time - start_time:.2f} seconds")

    def _estimate_gates(
        self,
        data,
        diff_matrix: jnp.ndarray | None = None,
        n_jobs: int | None = None,
        memory_limit: float = 0.8,
    ):
        if not self._fitted:
            ERROR("Attempting to estimate GATEs before model is fitted")
            raise RuntimeError("You must call .fit() prior to calling this method.")

        data = self.convert_dataframe_to_pandas(data, self.G)

        if self.G is None:
            DEBUG("No groups specified for GATE estimation")
            return

        if diff_matrix is None:
            diff_matrix = self._create_diff_matrix(data)

        INFO("Estimating Group Average Treatment Effects (GATEs)...")
        start_time = timeit.default_timer()

        groups = {group: data[group].unique() for group in self.G}

        total_combinations = sum(len(vals) for vals in groups.values())

        # Calculate optimal number of jobs if not specified
        if n_jobs is None:
            sample_size = data.shape[0]
            bytes_per_float = 8  # 64-bit floats
            estimated_memory_per_group = (
                sample_size * bytes_per_float * 3  # For mask, treated_mask, and results
                + diff_matrix.nbytes
                / total_combinations  # Portion of diff_matrix needed
            )
            n_jobs = self._calculate_optimal_jobs(
                memory_per_job=estimated_memory_per_group,
                total_memory_limit=memory_limit,
            )
            DEBUG(f"Using {n_jobs} parallel jobs")

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

        end_time = timeit.default_timer()
        DEBUG(f"GATE estimation completed in {end_time - start_time:.2f} seconds")

    def _create_diff_matrix(self, data) -> jnp.ndarray:
        DEBUG("Creating treatment difference matrix")
        start_time = timeit.default_timer()

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
        end_time = timeit.default_timer()
        DEBUG(
            f"Difference matrix created in {end_time - start_time:.2f} seconds. Shape: {diff.shape}"
        )
        return diff

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

    @staticmethod
    def _calculate_optimal_jobs(
        memory_per_job: float, total_memory_limit: float = 0.8
    ) -> int:
        cpu_cores = cpu_count()
        available_memory = psutil.virtual_memory().available
        memory_limit = int(available_memory * total_memory_limit)

        # Calculate jobs based on memory constraint
        memory_based_jobs = max(1, int(memory_limit // memory_per_job))

        # Calculate jobs based on CPU cores, leaving one core free
        cpu_based_jobs = max(1, cpu_cores - 1)

        # Take the minimum of memory and CPU constraints
        optimal_jobs = min(memory_based_jobs, cpu_based_jobs)

        DEBUG(
            f"Optimal jobs calculation: "
            f"memory_based={memory_based_jobs}, "
            f"cpu_based={cpu_based_jobs}, "
            f"selected={optimal_jobs}"
        )

        return optimal_jobs


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
