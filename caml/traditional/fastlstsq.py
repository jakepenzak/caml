import timeit

import flatdict
import pandas as pd
import patsy
from joblib import Parallel, delayed

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.stats as jstats

    _HAS_JAX = True
except ImportError:
    import numpy as jnp
    import scipy.stats as jstats

    _HAS_JAX = False


class FastLSTSQ:
    def __init__(
        self,
        Y: list[str],
        T: str,
        G: list[str] | None = None,
        X: list[str] | None = None,
        engine: str = "cpu",
    ):
        self.Y = Y
        self.T = T
        self.G = G
        self.X = X

        if engine not in ["cpu", "gpu"]:
            raise ValueError("Only 'cpu' and 'gpu' are supported for engine argument")

        if engine == "gpu":
            if not _HAS_JAX:
                raise ValueError("JAX is required for gpu engine.")
            try:
                len(jax.devices("gpu"))
            except RuntimeError:
                print("No available gpu, falling back to cpu.")
                engine = "cpu"

        self.engine = engine
        self.formula = self._create_formula(self.Y, self.T, self.G, self.X)
        self._fitted = False

    def fit_and_estimate(self, data, parallel=False):
        pd_df = self.convert_dataframe_to_pandas(data)
        for col in self.G or []:
            pd_df[col] = pd_df[col].astype("category")
        self.fit(pd_df)
        diff_matrix = self._create_diff_matrix(pd_df)
        self.estimate_ates(pd_df, diff_matrix)
        self.estimate_gates(pd_df, diff_matrix, parallel)

    def fit(self, data):
        data = self.convert_dataframe_to_pandas(data)

        self.results = flatdict.FlatDict({})

        start_time = timeit.default_timer()
        print("Converting data to design matrix...")
        y, X = patsy.dmatrices(self.formula, data=data)
        self._X_design_info = X.design_info
        end_time = timeit.default_timer()
        print(f"Done. Time: {end_time - start_time}")

        print("Putting data onto device...")
        start_time = timeit.default_timer()
        if _HAS_JAX:
            y = jnp.array(y, device=jax.devices(self.engine)[0])
            X = jnp.array(X, device=jax.devices(self.engine)[0])
        else:
            y = jnp.array(y)
            X = jnp.array(X)
        end_time = timeit.default_timer()
        print(f"Done. Time: {end_time - start_time}")

        print("Fitting regression model...")
        start_time = timeit.default_timer()

        @jax.jit
        def compute_params(X, y):
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

        params, vcv = compute_params(X, y)

        # params, _, _, _ = jnp.linalg.lstsq(X, y, rcond=None)
        # vcv = self._compute_vcv_matrix(y, X, params)

        self.results["params"] = params
        self.results["vcv"] = vcv
        self.results["std_err"] = jnp.sqrt(jnp.diagonal(vcv, axis1=1, axis2=2))
        self.results["treatment_effects"] = {}

        self._fitted = True
        end_time = timeit.default_timer()
        print(f"Done. Time: {end_time - start_time}")

    def estimate_ates(self, data, diff_matrix: jnp.ndarray | None = None):
        if not self._fitted:
            raise RuntimeError("You must call .fit() prior to calling this method.")

        data = self.convert_dataframe_to_pandas(data)

        if diff_matrix is None:
            diff_matrix = self._create_diff_matrix(data)

        print("Estimating Average Treatment Effects (ATEs)...")
        start_time = timeit.default_timer()
        treated_mask = jnp.array(data[self.T] == 1)
        statistics = self._compute_statistics(
            diff_matrix=diff_matrix,
            params=self.results["params"],
            vcv=self.results["vcv"],
            n_treated=diff_matrix[treated_mask].shape[0],
        )

        self.results["treatment_effects"]["overall"] = {}
        self.results["treatment_effects"]["overall"].update(
            {key: statistics[key] for key in statistics.keys()}
        )
        end_time = timeit.default_timer()
        print(f"Done. Time: {end_time - start_time}")

    def estimate_gates(
        self, data, diff_matrix: jnp.ndarray | None = None, parallel=False
    ):
        if not self._fitted:
            raise RuntimeError("You must call .fit() prior to calling this method.")

        data = self.convert_dataframe_to_pandas(data)

        if self.G is None:
            print("Note: No groups passed, therefore not GATEs will be estimated.")
        else:
            if diff_matrix is None:
                diff_matrix = self._create_diff_matrix(data)

            print("Estimating Group Average Treatment Effects (GATEs)...")
            start_time = timeit.default_timer()
            groups = {group: data[group].unique() for group in self.G}

            #################################################
            if parallel:
                group_info = []
                for group in groups:
                    for membership in groups[group]:
                        mask = jnp.array(data[group] == membership)
                        treated_mask = jnp.array(
                            data[data[group] == membership][self.T] == 1
                        )
                        group_key = f"{group}-{membership}"
                        group_info.append(
                            (group_key, group, membership, mask, treated_mask)
                        )

                params = self.results["params"]
                vcv = self.results["vcv"]

                # Process each group in parallel
                def process_group(group_key, mask, treated_mask):
                    diff_matrix_filtered = diff_matrix[mask]
                    n_treated = diff_matrix_filtered[treated_mask].shape[0]
                    statistics = self._compute_statistics(
                        diff_matrix=diff_matrix_filtered,
                        params=params,
                        vcv=vcv,
                        n_treated=n_treated,
                    )
                    return group_key, statistics

                # Run parallel processing
                results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(process_group)(group_key, mask, treated_mask)
                    for group_key, _, _, mask, treated_mask in group_info
                )

                # Update results dictionary
                for group_key, statistics in results:
                    self.results["treatment_effects"][group_key] = {}
                    self.results["treatment_effects"][group_key].update(
                        {key: statistics[key] for key in statistics.keys()}
                    )
            ###############################################
            else:
                group_treated_masks = {
                    (group, membership): (
                        jnp.array(data[group] == membership),
                        jnp.array(data[data[group] == membership][self.T] == 0),
                    )
                    for group in groups
                    for membership in groups[group]
                }
                for group, membership in group_treated_masks.keys():
                    diff_matrix_filtered = diff_matrix[
                        group_treated_masks[(group, membership)][0]
                    ]
                    treated_mask = group_treated_masks[(group, membership)][1]
                    statistics = self._compute_statistics(
                        diff_matrix=diff_matrix_filtered,
                        params=self.results["params"],
                        vcv=self.results["vcv"],
                        n_treated=diff_matrix_filtered[treated_mask].shape[0],
                    )

                    self.results["treatment_effects"][f"{group}-{membership}"] = {}
                    self.results["treatment_effects"][f"{group}-{membership}"].update(
                        {key: statistics[key] for key in statistics.keys()}
                    )

            end_time = timeit.default_timer()
            print(f"Done. Time: {end_time - start_time}")

    def estimate_cates(self):
        raise NotImplementedError("The estimate_cates method is not yet implemented.")

    def _create_diff_matrix(self, data) -> jnp.ndarray:
        print("Creating treatment difference matrix...")
        start_time = timeit.default_timer()
        data[f"{self.T}"] = 1
        X1 = patsy.dmatrix(self._X_design_info, data=data)
        data[f"{self.T}"] = 0
        X0 = patsy.dmatrix(self._X_design_info, data=data)

        if _HAS_JAX:
            X1 = jnp.array(X1, device=jax.devices(self.engine)[0])
            X0 = jnp.array(X0, device=jax.devices(self.engine)[0])
        else:
            X1 = jnp.array(X1)
            X0 = jnp.array(X0)

        diff = X1 - X0
        end_time = timeit.default_timer()
        print(f"Done. Time: {end_time - start_time}")
        return diff

    @staticmethod
    def _compute_statistics(
        diff_matrix: jnp.ndarray,
        params: jnp.ndarray,
        vcv: jnp.array,
        n_treated: jnp.array,
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
    def _compute_vcv_matrix(
        y: jnp.ndarray, X: jnp.ndarray, params: jnp.ndarray
    ) -> jnp.ndarray:
        y_resid = y - X @ params
        rss = jnp.sum(y_resid**2, axis=0)
        sigma_squared_hat = rss / (X.shape[0] - X.shape[1])
        XtX_inv = jnp.linalg.pinv(X.T @ X)

        vcv = (
            sigma_squared_hat[:, jnp.newaxis, jnp.newaxis] * XtX_inv[jnp.newaxis, :, :]
        )

        return vcv

    @staticmethod
    def _create_formula(
        Y: list[str], T: str, G: list[str], X: list[str] | None = None
    ) -> str:
        formula = ""
        for y in Y:
            formula += f"{y} "
            if y == Y[-1]:
                formula += "~"
            else:
                formula += "+ "

        formula += f" {T}"

        if G:
            for g in G:
                formula += f" + C({g})*{T}"

        if X:
            for x in X:
                formula += f" + {x}"

        return formula

    @staticmethod
    def convert_dataframe_to_pandas(dataframe) -> pd.DataFrame:
        if isinstance(dataframe, pd.DataFrame):
            return dataframe

        print("Converting input dataframe to Pandas...")

        if hasattr(dataframe, "toPandas"):
            return dataframe.toPandas()

        if hasattr(dataframe, "to_pandas"):
            return dataframe.to_pandas()

        raise Exception(
            f"Pandas conversion not currently supported for {type(dataframe)}."
        )
