from typing import Collection, Iterable

import pandas as pd
import patsy
from joblib import Parallel, delayed
from typeguard import typechecked

from ..generics import PandasConvertibleDataFrame, experimental, maybe_jit, timer
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
class FastOLS:
    r"""FastOLS is an optimized implementation of the OLS estimator designed specifically with treatment effect estimation in mind.

    **FastOLS is experimental and may change significantly in future versions.**

    This class estimates a standard linear regression model for any number of continuous or binary outcomes and a single continuous or binary treatment,
    and provides estimates for the Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs) out of the box. Additionally,
    methods are provided for estimating custom GATEs & Conditional Average Treatment Effects (CATEs) of individual observations, which can also be used for out-of-sample predictions.
    Note, this method assumes linear treatment effects and heterogeneity, which is typically sufficient when primarily concerned with ATEs and GATEs.

    This class leverages JAX for fast numerical computations, which can be installed using `pip install caml[jax]`, defaulting to NumPy if JAX is not
    available. For GPU acceleration, install JAX with GPU support using `pip install caml[jax-gpu]`.

    For outcome/treatment support, see [Support Matrix](support_matrix.qmd).

    For model specification details, see [Model Specifications](../02_Concepts/models.qmd#fastols).

    For a more detailed working example, see [FastOLS Example](../03_Examples/FastOLS.qmd).

    Parameters
    ----------
    Y : Collection[str]
        A list of outcome variable names.
    T : str
        The treatment variable name.
    G : Collection[str] | None
        A list of group variable names. These will be the groups for which GATEs will be estimated.
    X : Collection[str] | None
        A list of covariate variable names. These will be the covariates for which heterogeneity/CATEs can be estimated.
    W : Collection[str] | None
        A list of additional covariate variable names to be used as controls. These will be the additional covariates not used for modeling heterogeneity/CATEs.
    xformula : str | None
        Additional formula string to append to the main formula, starting with "+". For example, "+age+gender" will add age and gender as additional predictors.
    discrete_treatment : bool
        Whether the treatment is discrete
    engine : str
        The engine to use for computation. Can be "cpu" or "gpu". Note "gpu" requires JAX to be installed, which can be installed
        via `pip install caml[jax-gpu]`.

    Attributes
    ----------
    Y : Collection[str]
        A list of outcome variable names.
    T : str
        The treatment variable name.
    G : Collection[str] | None
        The list of group variable names. These will be the groups for which GATEs will be estimated.
    X : Collection[str] | None
        The list of variable names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATEs, that are in addition to G.
    W : Collection[str] | None
        The list of variable names representing the confounder/control feature **not** utilized for estimating heterogeneity/CATEs.
    formula : str
        The formula leveraged for design matrix creation via Patsy.
    params : np.ndarray
        The estimated parameters of the model.
    vcv : np.ndarray
        The estimated variance-covariance matrix of the model parameters.
    std_errors : np.ndarray
        The standard errors of the estimated parameters.
    treatment_effects : dict
        The estimated treatment effects dictionary.

    Examples
    --------
    ```{python}
    from caml import FastOLS
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data_generator = SyntheticDataGenerator(n_cont_outcomes=1,
                                                n_binary_outcomes=1,
                                                n_cont_modifiers=1,
                                                n_binary_modifiers=2,
                                                seed=10)
    df = data_generator.df

    fo_obj = FastOLS(
        Y=[c for c in df.columns if "Y" in c],
        T="T1_binary",
        G=[c for c in df.columns if "X" in c and ("bin" in c or "dis" in c)],
        X=[c for c in df.columns if "X" in c and "cont" in c],
        W=[c for c in df.columns if "W" in c],
        xformula=None,
        engine="cpu",
        discrete_treatment=True,
    )

    print(fo_obj)
    ```
    """

    def __init__(
        self,
        Y: Collection[str],
        T: str,
        G: Collection[str] | None = None,
        X: Collection[str] | None = None,
        W: Collection[str] | None = None,
        *,
        xformula: str | None = None,
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
        self._discrete_treatment = discrete_treatment

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

        self._engine = engine
        self.formula = self._create_formula(
            self.Y, self.T, self.G, self.X, self.W, self._discrete_treatment, xformula
        )
        self._formula = self.formula
        DEBUG(f"Created formula: {self.formula}")
        self._fitted = False
        self._treatment_effects = {}

    @property
    def params(self):
        if self._fitted:
            return self._params
        else:
            raise RuntimeError(
                "Model has not been fitted yet. Please run fit() method first."
            )

    @property
    def vcv(self):
        if self._fitted:
            return self._vcv
        else:
            raise RuntimeError(
                "Model has not been fitted yet. Please run fit() method first."
            )

    @property
    def std_err(self):
        if self._fitted:
            return self._std_err
        else:
            raise RuntimeError(
                "Model has not been fitted yet. Please run fit() method first."
            )

    @property
    def treatment_effects(self):
        if self._fitted:
            return self._treatment_effects
        else:
            raise RuntimeError(
                "Model has not been fitted yet. Please run fit() method first."
            )

    def fit(
        self,
        df: PandasConvertibleDataFrame,
        *,
        n_jobs: int = -1,
        estimate_effects: bool = True,
        robust_vcv: bool = False,
    ) -> None:
        """Fits the regression model on the provided data and, optionally, estimates Average Treatment Effect(s) (ATE) and Group Average Treatment Effect(s) (GATE).

        If `estimate_effects` is True, the method estimates Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs), based on specified `G`.
        This leverages `estimate_ate` method under the hood, but efficiently reuses the data and parallelizes the computation of GATEs.

        Parameters
        ----------
        df : PandasConvertibleDataFrame
            Input dataframe to fit the model on. Supported formats:
            pandas DataFrame, PySpark DataFrame, Polars DataFrame, or Any object with `toPandas()` or `to_pandas()` method
        n_jobs : int
            The number of jobs to use for parallel processing in the estimation of GATEs. Defaults to -1, which uses all available processors.
            If getting OOM errors, try setting n_jobs to a lower value.
        estimate_effects : bool
            Whether to estimate Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs).
        robust_vcv : bool
            Whether to use heteroskedasticity-robust (white) variance-covariance matrix and standard errors.

        Examples
        --------
        ```{python}
        fo_obj.fit(df, n_jobs=4, estimate_effects=True, robust_vcv=True)

        fo_obj.treatment_effects.keys()
        ```
        """
        pd_df = self._convert_dataframe_to_pandas(df, self.G)
        if self._discrete_treatment:
            if len(pd_df[self.T].unique()) != 2:
                raise ValueError("Treatment variable must be binary")
        y, X = self._create_design_matrix(pd_df)
        self._fit(X, y, robust_vcv=robust_vcv)
        self._fitted = True
        if estimate_effects:
            diff_matrix = self._create_design_matrix(pd_df, create_diff_matrix=True)
            self._treatment_effects = self.estimate_ate(
                pd_df,
                _diff_matrix=diff_matrix,
                return_results_dict=True,
                group="overall",
            )
            self._estimate_gates(pd_df, _diff_matrix=diff_matrix, n_jobs=n_jobs)

    @timer("ATE Estimation")
    def estimate_ate(
        self,
        df: PandasConvertibleDataFrame,
        *,
        return_results_dict: bool = False,
        group: str = "Custom Group",
        membership: str | None = None,
        _diff_matrix: jnp.ndarray | None = None,
    ) -> jnp.ndarray | dict:
        r"""Estimate Average Treatment Effects (ATEs) of `T` on each `Y` from fitted model.

        If the entire dataframe is provided, the function will estimate the ATE of the entire population, where the ATE, in the case of binary treatments, is formally defined as:
            $$
            \tau = \mathbb{E}_n[\mathbf{Y}_1 - \mathbf{Y}_0]
            $$

        If a subset of the dataframe is provided, the function will estimate the ATE of the subset (e.g., GATEs), where the GATE, in the case of binary treatments, is formally defined as:
            $$
            \tau = \mathbb{E}_n[\mathbf{Y}_1 - \mathbf{Y}_0|\mathbf{G}=G]
            $$

        For more details on treatment effect estimation, see [Model Specifications](../02_Concepts/models.qmd#treatment-effect-estimation-inference).

        Parameters
        ----------
        df : PandasConvertibleDataFrame
            Dataframe containing the data to estimate the ATEs. Supported formats:
            pandas DataFrame, PySpark DataFrame, Polars DataFrame, or Any object with `toPandas()` or `to_pandas()` method
        return_results_dict : bool
            If True, the function returns a dictionary containing ATEs/GATEs, standard errors, t-statistics, confidence intervals, and p-values.
            If False, the function returns a numpy array containing ATEs/GATEs alone.
        group : str
            Name of the group to estimate the ATEs for.
        membership : str | None
            Name of the membership variable to estimate the ATEs for.
        _diff_matrix : jnp.ndarray | None = None
            Private argument used in `fit` method.

        Returns
        -------
        jnp.ndarray | dict
            Estimated ATEs/GATEs or dictionary containing the estimated ATEs/GATEs and their standard errors, t-statistics, and p-values.

        Examples
        --------
        ```{python}
        ate = fo_obj.estimate_ate(df, return_results_dict=True, group="Overall")

        ate
        ```
        ```{python}
        df_filtered = df.query(
            "X3_binary == 0 & X1_continuous < 5"
        ).copy()

        custom_gate = fo_obj.estimate_ate(df_filtered)

        custom_gate
        ```
        """
        INFO("Estimating Average Treatment Effects (ATEs)...")

        if not self._fitted:
            raise RuntimeError("Model must be fitted before estimating ATEs.")

        if _diff_matrix is None:
            pd_df = self._convert_dataframe_to_pandas(df, self.G)
            diff_matrix = self._create_design_matrix(pd_df, create_diff_matrix=True)
        else:
            pd_df = df
            diff_matrix = _diff_matrix

        n_treated = int(pd_df[self.T].sum()) if self._discrete_treatment else None

        effects = self._compute_effects(
            diff_matrix=diff_matrix,
            params=self._params,
            vcv=self._vcv,
            n_treated=n_treated,
        )

        if return_results_dict:
            results = {}
            key = group if membership is None else f"{group}-{membership}"
            results[key] = {"outcome": self.Y}
            results[key].update(effects)
            return results

        return effects["ate"]

    @timer("CATE Estimation")
    def estimate_cate(
        self, df: PandasConvertibleDataFrame, *, return_results_dict: bool = False
    ) -> jnp.ndarray | dict:
        r"""Estimate Conditional Average Treatment Effects (CATEs) of `T` on each `Y` from fitted model for all given observations in the dataset.

        The CATE, in the case of binary treatments, is formally defined as:
            $$
            \tau = \mathbb{E}_n[\mathbf{Y}_1 - \mathbf{Y}_0|\mathbf{Q}=Q]
            $$

        For more details on treatment effect estimation, see [Model Specifications](../02_Concepts/models.qmd#treatment-effect-estimation-inference).

        Parameters
        ----------
        df : PandasConvertibleDataFrame
            Dataframe containing the data to estimate CATEs for. Supported formats:
                pandas DataFrame, PySpark DataFrame, Polars DataFrame, or Any object with `toPandas()` or `to_pandas()` method
        return_results_dict : bool
            If True, the function returns a dictionary containing CATEs, standard errors, t-statistics, confidence intervals, and p-values.
            If False, the function returns a numpy array containing CATEs alone.

        Returns
        -------
        jnp.ndarray | dict
            CATEs or dictionary containing CATEs, standard errors, t-statistics, and p-values.

        Examples
        --------
        ```{python}
        cates = fo_obj.estimate_cate(df)
        cates[:5]
        ```
        ```{python}
        res = fo_obj.estimate_cate(df, return_results_dict=True)
        res.keys()
        ```
        """
        INFO("Estimating Conditional Average Treatment Effects (CATEs)...")

        if not self._fitted:
            raise RuntimeError("Model must be fitted before estimating ATEs.")

        pd_df = self._convert_dataframe_to_pandas(df, self.G)
        diff_matrix = self._create_design_matrix(pd_df, create_diff_matrix=True)

        statistics = self._compute_effects(
            diff_matrix, self._params, self._vcv, is_cates=True
        )

        if return_results_dict:
            results = {"outcome": self.Y}
            results.update(statistics)
            return results
        return statistics["cate"]

    def predict(
        self,
        df: PandasConvertibleDataFrame,
        *,
        return_results_dict: bool = False,
        mode: str = "cate",
    ) -> jnp.ndarray | dict:
        """Generate predicted conditional average treatment effects (CATEs) or outcomes.

        When mode is "outcome", the function returns predicted outcomes.

        When mode is "cate", the function returns predicted CATEs, behaving as an alias for `estimate_cate`.

        Parameters
        ----------
        df : PandasConvertibleDataFrame
            Dataframe containing the data to estimate CATEs for. Supported formats:
                pandas DataFrame, PySpark DataFrame, Polars DataFrame, or Any object with `toPandas()` or `to_pandas()` method
        return_results_dict : bool
            If True, the function returns a dictionary containing CATEs, standard errors, t-statistics, confidence intervals, and p-values.
            If False, the function returns a numpy array containing CATEs alone.
            Does not have any effect when mode is "outcome".
        mode : str
            The mode of prediction. Supported modes are "cate" and "outcome".
            If "cate", the function returns CATEs.
            If "outcome", the function returns predicted outcomes.

        Returns
        -------
        jnp.ndarray | dict
            CATEs or dictionary containing CATEs, standard errors, t-statistics, and p-values.

        Examples
        --------
        ```{python}
        cates = fo_obj.predict(df)
        cates[:5]
        ```
        ```{python}
        res = fo_obj.predict(df, return_results_dict=True)
        res.keys()
        ```
        """
        if mode == "cate":
            return self.estimate_cate(df, return_results_dict=return_results_dict)
        elif mode == "outcome":
            _, X = self._create_design_matrix(df)
            return X @ self.params
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be either 'cate' or 'outcome'."
            )

    def prettify_treatment_effects(self, effects: dict | None = None) -> pd.DataFrame:
        """Convert treatment effects dictionary to a pandas DataFrame.

        If no argument is provided, the results are constructed from internal results dictionary. This is
        useful default behavior. For custom treatment effects, you can pass the results generated
        by the `estimate_ate` method.

        Parameters
        ----------
        effects : dict, optional
            Dictionary of treatment effects. If None, the results are constructed from internal results dictionary.

        Returns
        -------
        pd.DataFrame
            DataFrame of treatment effects.

        Examples
        --------
        ```{python}
        fo_obj.prettify_treatment_effects()
        ```
        ```{python}
        ## Using a custom GATE
        custom_gate = fo_obj.estimate_ate(df_filtered, return_results_dict=True, group="My Custom Group")
        fo_obj.prettify_treatment_effects(custom_gate)
        ```
        """
        if effects is None:
            effects = self._treatment_effects
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
                    if isinstance(value, list):
                        final_results[stat] = value.copy()
                    elif isinstance(value, jnp.ndarray):
                        final_results[stat] = value.flatten().copy()
                    elif isinstance(value, int):
                        final_results[stat] = [value] * n_outcomes
            else:
                final_results["group"] += [group] * n_outcomes
                final_results["membership"] += [membership] * n_outcomes
                for stat, value in effects[k].items():
                    if isinstance(value, list):
                        final_results[stat] += value
                    elif isinstance(value, jnp.ndarray):
                        final_results[stat] = jnp.hstack(
                            [final_results[stat], value.flatten()]
                        )
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

        self._params = params
        self._vcv = vcv
        self._std_err = jnp.sqrt(jnp.diagonal(vcv, axis1=1, axis2=2)).T
        self._treatment_effects = {}

    @timer("Design Matrix Creation")
    def _create_design_matrix(
        self, df: pd.DataFrame, create_diff_matrix: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        try:
            if not create_diff_matrix:
                DEBUG("Creating model matrix...")
                y, X = patsy.dmatrices(self.formula, data=df, NA_action="raise")

                self._X_design_info = X.design_info

                if _HAS_JAX:
                    y = jnp.array(y, device=jax.devices(self._engine)[0])
                    X = jnp.array(X, device=jax.devices(self._engine)[0])
                else:
                    y = jnp.array(y)
                    X = jnp.array(X)

                return y, X
            else:
                DEBUG("Creating treatment difference matrix...")
                original_t = df[self.T].copy()

                if self._discrete_treatment:
                    df[self.T] = 0
                    X0 = patsy.dmatrix(self._X_design_info, data=df, NA_action="raise")
                    df[self.T] = 1
                    X1 = patsy.dmatrix(self._X_design_info, data=df)
                else:
                    X0 = patsy.dmatrix(self._X_design_info, data=df)
                    df[self.T] = df[self.T] + 1
                    X1 = patsy.dmatrix(self._X_design_info, data=df)

                df[self.T] = original_t

                if _HAS_JAX:
                    X1 = jnp.array(X1, device=jax.devices(self._engine)[0])
                    X0 = jnp.array(X0, device=jax.devices(self._engine)[0])
                else:
                    X1 = jnp.array(X1)
                    X0 = jnp.array(X0)

                diff = X1 - X0

                return diff
        except patsy.PatsyError as e:
            if "factor contains missing values" in str(e):
                raise ValueError(
                    "Input DataFrame contains missing values. Please handle missing values before proceeding."
                )

    @staticmethod
    def _compute_effects(
        diff_matrix: jnp.ndarray,
        params: jnp.ndarray,
        vcv: jnp.ndarray,
        n_treated: int | None = None,
        is_cates: bool = False,
    ) -> dict:
        if is_cates:
            d = diff_matrix
        else:
            d = jnp.mean(diff_matrix, axis=0).reshape(1, -1)

        effect = d @ params
        std_err = jnp.sqrt(jnp.einsum("nj,ojk,nk->no", d, vcv, d))
        t_stat = jnp.where(std_err > 0, effect / std_err, 0)
        pval = 2 * (1 - jstats.norm.cdf(jnp.abs(t_stat)))
        n = diff_matrix.shape[0]
        if n_treated is not None:
            n_control = n - n_treated
        else:
            n_control = None

        if is_cates:
            return {
                "cate": effect,
                "std_err": std_err,
                "t_stat": t_stat,
                "pval": pval,
            }
        else:
            return {
                "ate": effect,
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
        df: pd.DataFrame,
        *,
        n_jobs: int = -1,
        _diff_matrix: jnp.ndarray,
    ):
        if self.G is None:
            DEBUG("No groups specified for GATE estimation. Skipping.")
            return

        INFO("Estimating Group Average Treatment Effects (GATEs)...")

        groups = {group: df[group].unique() for group in self.G}

        # Prepare groups for processing
        group_info = []
        for group in groups:
            for membership in groups[group]:
                mask = jnp.array(df[group] == membership)
                treated_mask = (
                    jnp.array(df[df[group] == membership][self.T] == 1)
                    if self._discrete_treatment
                    else None
                )
                group_key = f"{group}-{membership}"
                group_info.append((group_key, mask, treated_mask))

        def process_group(group_key, mask, treated_mask):
            diff_matrix_filtered = _diff_matrix[mask]
            n_treated = int(treated_mask.sum()) if self._discrete_treatment else None
            statistics = self._compute_effects(
                diff_matrix=diff_matrix_filtered,
                params=self._params,
                vcv=self._vcv,
                n_treated=n_treated,
            )
            return group_key, statistics

        DEBUG(f"Starting parallel processing with {n_jobs} jobs")
        results: Iterable = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_group)(group_key, mask, treated_mask)
            for group_key, mask, treated_mask in group_info
        )

        for group_key, statistics in results:
            self._treatment_effects[group_key] = {"outcome": self.Y}
            self._treatment_effects[group_key].update(statistics)

    @staticmethod
    def _convert_dataframe_to_pandas(df, groups) -> pd.DataFrame:
        def convert_groups_to_categorical(df, groups):
            for col in groups or []:
                df[col] = df[col].astype("category")
            return df

        if isinstance(df, PandasConvertibleDataFrame):
            if isinstance(df, pd.DataFrame):
                return convert_groups_to_categorical(df, groups)

            DEBUG(f"Converting input dataframe of type {type(df)} to pandas")
            if hasattr(df, "toPandas"):
                return convert_groups_to_categorical(df.toPandas(), groups)
            if hasattr(df, "to_pandas"):
                return convert_groups_to_categorical(df.to_pandas(), groups)

        ERROR(f"Unsupported dataframe type: {type(df)}")
        raise ValueError(f"Pandas conversion not currently supported for {type(df)}.")

    @staticmethod
    def _create_formula(
        Y: Collection[str],
        T: str,
        G: Collection[str] | None = None,
        X: Collection[str] | None = None,
        W: Collection[str] | None = None,
        discrete_treatment: bool = False,
        xformula: str | None = None,
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

        if xformula:
            formula += f" {xformula}"

        return formula

    def __str__(self):
        """
        Returns a string representation of the FastOLS object.

        Returns
        -------
        str
            A string containing information about the FastOLS object.
        """
        summary = (
            "================== FastOLS Object ==================\n"
            + f"Engine: {self._engine}\n"
            + f"Outcome Variable: {self.Y}\n"
            + f"Treatment Variable: {self.T}\n"
            + f"Discrete Treatment: {self._discrete_treatment}\n"
            + f"Group Variables: {self.G}\n"
            + f"Features/Confounders for Heterogeneity (X): {self.X}\n"
            + f"Features/Confounders as Controls (W): {self.W}\n"
            + f"Formula: {self.formula}\n"
        )

        return summary
