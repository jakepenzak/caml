from typing import Any, Collection, Iterable

import pandas as pd
import patsy
from joblib import Parallel, delayed
from typeguard import typechecked

from .._generics import experimental, maybe_jit, timer
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

DataFrameLike = Any


@experimental
@typechecked
class FastOLS:
    r"""FastOLS is a fast implementation of the OLS estimator designed specifically with treatment effect estimation in mind.

    **FastOLS is experimental and may change significantly in future versions.**

    This class estimates a standard linear regression model for any number of continuous or binary outcomes and a single continuous or binary treatment,
    and provides estimates for the Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs) out of the box. Additionally,
    methods are provided for estimating & predicting Conditional Average Treatment Effects (CATEs) of individual observations and custom GATEs
    can also be estimated. Note, this method assumes linear treatment effects and heterogeneity. This is typically sufficient when primarily concerned with
    average treatment effects.

    This class leverages JAX for fast numerical computations, which can be installed using `pip install caml[jax]`, defaulting to NumPy if JAX is not
    available. For GPU acceleration, install JAX with GPU support using `pip install caml[jax-gpu]`.

    For outcome/treatment support, see [matrix](support_matrix.qmd).

    ## Model Specification

    The model is given by:
    $$
    \begin{equation}
    \mathbf{Y} = T \beta + \mathbf{Q}\mathbf{\Gamma} + \left(T \circ \mathbf{Q}\right)\mathbf{\Omega} + \mathbf{W}\mathbf{\Psi} + \mathbf{E}
    \tag{1}
    \end{equation}
    $$

    where $\mathbf{Y}_{n \times p}$ is the matrix of $p$ outcomes, $T_{n \times 1}$ is the treatment variable,
    $\mathbf{Q}_{n \times (j+l)} = \bigl[\mathbf{X} \; \mathbf{G} \bigr]$ is the horizontal stack matrix of $j$ covariates and $l$ group variables,
    $\mathbf{W}_{n \times m}$ is the matrix of $m$ control covariates, $\beta_{1 \times p}$ is the vector of coefficients on $T$,
    $\mathbf{\Gamma}_{(j+l) \times p}$ is the matrix of coefficients on $\mathbf{Q}$, $\mathbf{\Omega}_{(j+l) \times p}$ is the matrix
    of coefficients on the interaction terms between $T$ and $\mathbf{Q}$, $\mathbf{\Psi}_{m \times p}$ is the matrix of
    coefficients on $\mathbf{W}$, and $\mathbf{E}_{n \times p}$ is the error term matrix.

    $\mathbf{Q}$ contains the covariates and group variables used to model treatment effect heterogeneity via interaction terms.

    #### Treatment Effect Estimation & Inference

    Our average treatment effect (ATE) $\tau$ for a binary treatment variable $T$ is defined as:

    $$
    \tau = \mathbb{E}_n\left[\mathbb{E}\left[\mathbf{Y} \mid T = 1\right] - \mathbb{E}\left[\mathbf{Y} \mid T = 0\right]\right]
    $$

    Let $D$ denote the design matrix for (1), then assuming exogeneity in $T$, the ATEs are identified and can be estimated as follows:
    $$
    \mathbf{\tau} = \mathbf{\Theta'}\bar{d}
    $$

    where $\mathbf{\Theta'} = \left[\beta' \; \mathbf{\Gamma'} \; \mathbf{\Omega'} \; \mathbf{\Psi'}\right]$ is
    the horizontally concatenated matrix of transposed coefficient matrices, and
    $\bar{d} = \mathbb{E}_n\left[D_{T=1} - D_{T=0}\right]$ is the the average difference in the design matrix for all observations.

    Furthermore, for each outcome $k \in \{1,2,...,p\}$, we can estimate the standard error of the ATE as follows:
    $$
    \text{SE}(\tau_k) = \sqrt{\bar{d}'\text{VCV}(\mathbf{\Theta}_k)\bar{d}}
    $$

    where $\text{VCV}(\mathbf{\Theta}_k)$ is the variance-covariance matrix of the estimated coefficients for the $k$-th outcome.

    This logic extends naturally to the estimation of GATEs and CATEs, where $\bar{d} = \mathbb{E}_n\left[D_{T=1} - D_{T=0} | \mathbf{G}=g\right]$
    $\bar{d} = \mathbb{E}_n\left[D_{T=1} - D_{T=0} | \mathbf{G}=g, \mathbf{X}=x\right]$, $\dots$, etc. and to continuous treatments
    $\bar{d} = \mathbb{E}_n\left[D_{T=t+1} - D_{T=t}\right]$, $\dots$, etc.

    Parameters
    ----------
    Y : Collection[str]
        A list of outcome variable names.
    T : str
        The treatment variable name.
    G : Collection[str] | None
        A list of group variable names, by default None. These will be the groups for which GATEs will be estimated.
    X : Collection[str] | None
        A list of covariate variable names, by default None. These will be the covariates for which heterogeneity/CATEs can be estimated.
    W : Collection[str] | None
        A list of additional covariate variable names to be used as controls, by default None. These will be the additional covariates not used for modeling heterogeneity/CATEs.
    discrete_treatment : bool
        Whether the treatment is discrete, by default False
    engine : str
        The engine to use for computation, by default "cpu". Can be "cpu" or "gpu". Note "gpu" requires JAX to be installed, which can be installed
        via `pip install caml[jax-gpu]`.

    Attributes
    ----------
    Y : Collection[str]
        A list of outcome variable names.
    T : str
        The treatment variable name.
    G : Collection[str] | None
        The list of group variable names, by default None. These will be the groups for which GATEs will be estimated.
    X : Collection[str] | None
        The list of variable names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATEs, that are in addition to G.
    W : Collection[str] | None
        The list of variable names representing the confounder/control feature **not** utilized for estimating heterogeneity/CATEs.
    discrete_treatment : bool
        Whether the treatment is discrete, by default True
    engine : str
        The engine to use for computation, by default "cpu". Can be "cpu" or "gpu". Note "gpu" requires JAX to be installed, which can be installed
        via `pip install caml[jax-gpu]`
    formula : str
        The formula leveraged for design matrix creation via Patsy.
    results : dict
        A dictionary containing the results of the fitted model & estimated ATEs/GATEs.
    """

    def __init__(
        self,
        Y: Collection[str],
        T: str,
        G: Collection[str] | None = None,
        X: Collection[str] | None = None,
        W: Collection[str] | None = None,
        *,
        discrete_treatment: bool = False,
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
        self.formula = self._create_formula(
            self.Y, self.T, self.G, self.X, self.W, self.discrete_treatment
        )
        DEBUG(f"Created formula: {self.formula}")
        self._fitted = False
        self.results = {}

    def fit(
        self,
        data: DataFrameLike,
        *,
        n_jobs: int = -1,
        estimate_effects: bool = True,
        robust_vcv: bool = False,
    ):
        """
        Fits the regression model on the provided data and estimates ATEs and GATEs.

        Parameters
        ----------
        data : DataFrameLike
            Input data to fit the model on. Supported formats:
            pandas DataFrame, PySpark DataFrame, Polars DataFrame, or Any object with toPandas() or to_pandas() method

        n_jobs : int
            The number of jobs to use for parallel processing in the estimation of GATEs. Defaults to -1, which uses all available processors.
            If getting OOM errors, try setting n_jobs to a lower value.

        estimate_effects : bool
            Whether to estimate Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs).

        robust_vcv : bool
            Whether to use heteroskedasticity-robust (white) variance-covariance matrix and standard errors.
        """
        pd_df = self._convert_dataframe_to_pandas(data, self.G)
        if self.discrete_treatment:
            if len(pd_df[self.T].unique()) != 2:
                raise ValueError("Treatment variable must be binary")
        y, X = self._create_design_matrix(pd_df)
        self._fit(X, y, robust_vcv=robust_vcv)
        if estimate_effects:
            diff_matrix = self._create_design_matrix(pd_df, create_diff_matrix=True)
            self.results["treatment_effects"] = self.estimate_ate(
                pd_df, _diff_matrix=diff_matrix, group="overall", membership=""
            )
            self._estimate_gates(pd_df, _diff_matrix=diff_matrix, n_jobs=n_jobs)
        self._fitted = True

    @timer("ATE Estimation")
    def estimate_ate(
        self,
        data: DataFrameLike,
        *,
        group: str = "Custom Group",
        membership: str | None = None,
        _diff_matrix: jnp.ndarray | None = None,
    ) -> dict:
        INFO("Estimating Average Treatment Effects (ATEs)...")

        if _diff_matrix is None:
            data = self._convert_dataframe_to_pandas(data, self.G)
            diff_matrix = self._create_design_matrix(data, create_diff_matrix=True)
        else:
            diff_matrix = _diff_matrix

        n_treated = int(data[self.T].sum()) if self.discrete_treatment else None

        statistics = self._compute_statistics(
            diff_matrix=diff_matrix,
            params=self.results["params"],
            vcv=self.results["vcv"],
            n_treated=n_treated,
        )

        results = {}
        results[f"{group}-{membership}"] = {"outcome": self.Y}
        results[f"{group}-{membership}"].update(
            {key: statistics[key] for key in statistics.keys()}
        )
        return results

    @timer("CATE Estimation")
    def estimate_cate(self) -> jnp.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support estimate_cates method yet."
        )

    def predict(self) -> jnp.ndarray:
        return self.estimate_cate()

    def prettify_treatment_effects(self, effects: dict | None = None) -> pd.DataFrame:
        if effects is None:
            effects = self.results["treatment_effects"]
        n_outcomes = len(self.Y)

        final_results = {}
        for i, k in enumerate(effects.keys()):
            try:
                group = k.split("-")[0]
                membership = k.split("-")[1]
            except IndexError:
                group = k
                membership = None
            if i == 0:
                final_results["group"] = [group] * n_outcomes
                final_results["membership"] = [membership] * n_outcomes
                for stat, value in effects[k].items():
                    if isinstance(value, list) or isinstance(value, jnp.ndarray):
                        final_results[stat] = value.copy()
                    elif isinstance(value, int):
                        final_results[stat] = [value] * n_outcomes
            else:
                final_results["group"] += [group] * n_outcomes
                final_results["membership"] += [membership] * n_outcomes
                for stat, value in effects[k].items():
                    if isinstance(value, list):
                        final_results[stat] += value
                    elif isinstance(value, jnp.ndarray):
                        final_results[stat] = jnp.hstack([final_results[stat], value])
                    elif isinstance(value, int):
                        final_results[stat] += [value] * n_outcomes

        return pd.DataFrame(final_results)

    @timer("Model Fitting")
    def _fit(self, X: jnp.ndarray, y: jnp.ndarray, robust_vcv: bool = False):
        INFO("Fitting regression model...")

        @maybe_jit
        def fit(X, y):
            params, _, _, _ = jnp.linalg.lstsq(X, y, rcond=None)
            y_resid = y - X @ params
            XtX_inv = jnp.linalg.pinv(X.T @ X)
            if robust_vcv:
                E = y_resid**2
                XEX = jnp.einsum("ni,nj,no->oij", X, X, E)
                vcv = XtX_inv @ XEX @ XtX_inv
            else:
                rss = jnp.sum(y_resid**2, axis=0)
                sigma_squared_hat = rss / (X.shape[0] - X.shape[1])
                XtX_inv = jnp.linalg.pinv(X.T @ X)
                vcv = jnp.einsum("o,ij->oij", sigma_squared_hat, XtX_inv)
            return params, vcv

        params, vcv = fit(X, y)

        self.results["params"] = params
        self.results["vcv"] = vcv
        self.results["std_err"] = jnp.sqrt(jnp.diagonal(vcv, axis1=1, axis2=2))
        self.results["treatment_effects"] = {}

    @timer("Design Matrix Creation")
    def _create_design_matrix(
        self, data: pd.DataFrame, create_diff_matrix: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        if create_diff_matrix:
            DEBUG("Creating treatment difference matrix...")
            original_t = data[self.T].copy()

            if self.discrete_treatment:
                data[self.T] = 0
                X0 = patsy.dmatrix(self._X_design_info, data=data)
                data[self.T] = 1
                X1 = patsy.dmatrix(self._X_design_info, data=data)
            else:
                X0 = patsy.dmatrix(self._X_design_info, data=data)
                data[self.T] = data[self.T] + 1
                X1 = patsy.dmatrix(self._X_design_info, data=data)

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
    def _compute_statistics(
        diff_matrix: jnp.ndarray,
        params: jnp.ndarray,
        vcv: jnp.ndarray,
        n_treated: int | None,
    ) -> dict:
        diff_mean = jnp.mean(diff_matrix, axis=0)

        ate = diff_mean @ params
        std_err = jnp.sqrt(diff_mean @ vcv @ diff_mean.T)
        t_stat = jnp.where(std_err > 0, ate / std_err, 0)
        pval = 2 * (1 - jstats.norm.cdf(jnp.abs(t_stat)))
        n = diff_matrix.shape[0]
        if n_treated is not None:
            n_control = n - n_treated
        else:
            n_control = None

        return {
            "ate": ate,
            "std_err": std_err,
            "t_stat": t_stat,
            "pval": pval,
            "n": n,
            "n_treated": n_treated,
            "n_control": n_control,
        }

    @timer("Prespecified GATE Estimation")
    def _estimate_gates(
        self,
        data: DataFrameLike,
        *,
        n_jobs: int = -1,
        _diff_matrix: jnp.ndarray,
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
                treated_mask = (
                    jnp.array(data[data[group] == membership][self.T] == 1)
                    if self.discrete_treatment
                    else None
                )
                group_key = f"{group}-{membership}"
                group_info.append((group_key, mask, treated_mask))

        params = self.results["params"]
        vcv = self.results["vcv"]

        def process_group(group_key, mask, treated_mask):
            diff_matrix_filtered = _diff_matrix[mask]
            n_treated = int(treated_mask.sum()) if self.discrete_treatment else None
            statistics = self._compute_statistics(
                diff_matrix=diff_matrix_filtered,
                params=params,
                vcv=vcv,
                n_treated=n_treated,
            )
            return group_key, statistics

        DEBUG(f"Starting parallel processing with {n_jobs} jobs")
        results: Iterable = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_group)(group_key, mask, treated_mask)
            for group_key, mask, treated_mask in group_info
        )

        for group_key, statistics in results:
            self.results["treatment_effects"][group_key] = {"outcome": self.Y}
            self.results["treatment_effects"][group_key].update(
                {key: statistics[key] for key in statistics.keys()}
            )

    @staticmethod
    def _convert_dataframe_to_pandas(dataframe, groups) -> pd.DataFrame:
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
    def _create_formula(
        Y: Collection[str],
        T: str,
        G: Collection[str] | None = None,
        X: Collection[str] | None = None,
        W: Collection[str] | None = None,
        discrete_treatment: bool = False,
    ) -> str:
        formula = " + ".join(Y)

        if discrete_treatment:
            treatment = f"C({T})"
        else:
            treatment = T

        formula += f" ~ {treatment}"

        for g in G or []:
            formula += f" + C({g})*{treatment}"

        for x in X or []:
            formula += f" + {x}*{treatment}"

        for w in W or []:
            formula += f" + {w}"

        return formula
