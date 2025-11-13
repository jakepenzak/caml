from typing import Any, NoReturn, Sequence

import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
from joblib import Parallel, delayed

from caml.core._base import BaseCamlEstimator
from caml.generics.decorators import experimental, timer
from caml.generics.interfaces import FittedAttr, PandasConvertibleDataFrame
from caml.generics.logging import DEBUG, INFO


@experimental
class InteractiveLinearRegression(BaseCamlEstimator):
    r"""InteractiveLinearRegression is an interactive linear regression estimator with explicit treatment interaction terms, enabling precision improvements & heterogeneous treatment discovery.

    **InteractiveLinearRegression is experimental and may change significantly in future versions.**

    This class estimates a standard linear regression model, with treatment-covariate interaction terms, for any number of continuous or binary outcomes and a single continuous or binary treatment,
    and provides estimates for the Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs) out of the box. Additionally,
    methods are provided for estimating custom GATEs & Conditional Average Treatment Effects (CATEs) of individual observations, which can also be used for out-of-sample predictions.
    Note, this method assumes linear treatment effects and heterogeneity, which is typically sufficient when primarily concerned with ATEs and GATEs.

    This model is designed and adapted

    For outcome/treatment support, see [Support Matrix](support_matrix.qmd).

    For model specification details, see [Model Specifications](../02_Concepts/models.qmd#interactivelinearregression).

    For a more detailed working example, see [InteractiveLinearRegression Example](../03_Examples/InteractiveLinearRegression.qmd).

    Parameters
    ----------
    Y : Sequence[str]
        A list of outcome variable names.
    T : str
        The treatment variable name.
    G : Sequence[str] | None
        A list of group (categorical) variable names, used for interaction terms. These will be the groups for which GATEs will be estimated.
    X : Sequence[str] | None
        A list of non-categorical covariate variable names, used for interaction terms. These will be additional covariates for which heterogeneity/CATEs can be estimated.
    W : Sequence[str] | None
        A list of additional covariate variable names to be used as controls, not interacted with the treatment. These will be the additional covariates not used for modeling heterogeneity/CATEs.
    xformula : str | None
        Additional formula string to append to the main formula, starting with "+". For example, "+age+gender" will add age and gender as additional predictors.
    discrete_treatment : bool
        Whether the treatment is discrete

    Attributes
    ----------
    Y : Sequence[str]
        A list of outcome variable names.
    T : str
        The treatment variable name.
    G : Sequence[str] | None
        The list of group variable names. These will be the groups for which GATEs will be estimated.
    X : Sequence[str] | None
        The list of variable names representing the confounder/control feature set to be utilized for estimating heterogeneity/CATEs, that are in addition to G.
    W : Sequence[str] | None
        The list of variable names representing the confounder/control feature **not** utilized for estimating heterogeneity/CATEs.
    formula : str
        The formula leveraged for design matrix creation via Patsy.
    params : np.ndarray
        The estimated parameters of the model.
    vcv : np.ndarray
        The estimated variance-covariance matrix of the model parameters.
    std_err : np.ndarray
        The standard errors of the estimated parameters.
    fitted_values : np.ndarray
        The predicted values from the model.
    residuals : np.ndarray
        The residuals of the model.
    treatment_effects : dict
        The estimated treatment effects dictionary.

    Examples
    --------
    ```{python}
    from caml import InteractiveLinearRegression
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data_generator = SyntheticDataGenerator(n_cont_outcomes=1,
                                                n_binary_outcomes=1,
                                                n_cont_modifiers=1,
                                                n_binary_modifiers=2,
                                                seed=10)
    df = data_generator.df

    ilr = InteractiveLinearRegression(
        Y=[c for c in df.columns if "Y" in c],
        T="T1_binary",
        G=[c for c in df.columns if "X" in c and ("bin" in c or "dis" in c)],
        X=[c for c in df.columns if "X" in c and "cont" in c],
        W=[c for c in df.columns if "W" in c],
        xformula=None,
        discrete_treatment=True,
    )

    print(ilr)
    ```
    """

    params = FittedAttr("_params")
    vcv = FittedAttr("_vcv")
    std_err = FittedAttr("_std_err")
    treatment_effects = FittedAttr("_treatment_effects")
    fitted_values = FittedAttr("_fitted_values")
    residuals = FittedAttr("_residuals")

    def __init__(
        self,
        Y: Sequence[str],
        T: str,
        G: Sequence[str] | None = None,
        X: Sequence[str] | None = None,
        W: Sequence[str] | None = None,
        *,
        xformula: str | None = None,
        discrete_treatment: bool = True,
    ):
        DEBUG(
            f"Initializing {self.__class__.__name__} with parameters: Y={Y}, T={T}, G={G}, X={X}, W={W}, discrete_treatment={discrete_treatment}"
        )
        self.Y = list(Y)
        self.T = T
        self.G = list(G) if G else list()
        self.X = list(X) if X else list()
        self.W = list(W) if W else list()
        self._discrete_treatment = discrete_treatment

        self.formula = self._create_formula(
            self.Y, self.T, self.G, self.X, self.W, self._discrete_treatment, xformula
        )
        self._formula = self.formula
        DEBUG(f"Created formula: {self.formula}")
        self._fitted = False
        self._treatment_effects: dict = {}

    def fit(
        self,
        df: PandasConvertibleDataFrame,
        *,
        n_jobs: int = -1,
        estimate_effects: bool = True,
        cov_type: str = "nonrobust",
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
        cov_type : str
            The covariance estimator to use for variance-covariance matrix and standard errors. Can be "nonrobust", "HC0", or "HC1".

        Examples
        --------
        ```{python}
        ilr.fit(df, n_jobs=4, estimate_effects=True, cov_type='nonrobust')

        ilr.treatment_effects.keys()
        ```
        """
        if cov_type not in ("nonrobust", "HC0", "HC1"):
            raise ValueError("cov_type must be 'nonrobust', 'HC0', or 'HC1'")

        pd_df = self._convert_dataframe_to_pandas(df, self.G)
        if self._discrete_treatment:
            if len(pd_df[self.T].unique()) != 2:
                raise ValueError("Treatment variable must be binary")
        y, X = self._create_design_matrix(pd_df)
        self._fit(X, y, cov_type=cov_type)
        self._fitted = True
        if estimate_effects:
            diff_matrix = self._create_difference_matrix(pd_df)
            self._treatment_effects = self.estimate_ate(  # pyright: ignore[reportAttributeAccessIssue]
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
        _diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray | dict:
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
            If True, the function returns a dictionary containing ATEs/GATEs, standard errors, t-statistics, and p-values.
            If False, the function returns a numpy array containing ATEs/GATEs alone.
        group : str
            Name of the group to estimate the ATEs for.
        membership : str | None
            Name of the membership variable to estimate the ATEs for.
        _diff_matrix : np.ndarray | None = None
            Private argument used in `fit` method.

        Returns
        -------
        np.ndarray | dict
            Estimated ATEs/GATEs or dictionary containing the estimated ATEs/GATEs and their standard errors, t-statistics, and p-values.

        Examples
        --------
        ```{python}
        ate = ilr.estimate_ate(df, return_results_dict=True, group="Overall")

        ate
        ```
        ```{python}
        df_filtered = df.query(
            "X3_binary == 0 & X1_continuous < 5"
        ).copy()

        custom_gate = ilr.estimate_ate(df_filtered)

        custom_gate
        ```
        """
        INFO("Estimating Average Treatment Effects (ATEs)...")

        if not self._fitted:
            raise RuntimeError("Model must be fitted before estimating ATEs.")

        pd_df = self._convert_dataframe_to_pandas(df, self.G)
        if _diff_matrix is None:
            diff_matrix = self._create_difference_matrix(pd_df)
        else:
            diff_matrix = _diff_matrix

        n_treated = int(pd_df[self.T].sum()) if self._discrete_treatment else None

        effects = self._compute_effects(
            diff_matrix=diff_matrix,
            params=self._params,
            vcv=self._vcv,
            n_treated=n_treated,
            include_inference=return_results_dict,
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
    ) -> np.ndarray | dict:
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
            If True, the function returns a dictionary containing CATEs, standard errors, t-statistics, and p-values.
            If False, the function returns a numpy array containing CATEs alone.

        Returns
        -------
        np.ndarray | dict
            CATEs or dictionary containing CATEs, standard errors, t-statistics, and p-values.

        Examples
        --------
        ```{python}
        cates = ilr.estimate_cate(df)
        cates[:5]
        ```
        ```{python}
        res = ilr.estimate_cate(df, return_results_dict=True)
        res.keys()
        ```
        """
        INFO("Estimating Conditional Average Treatment Effects (CATEs)...")

        if not self._fitted:
            raise RuntimeError("Model must be fitted before estimating ATEs.")

        pd_df = self._convert_dataframe_to_pandas(df, self.G)
        diff_matrix = self._create_difference_matrix(pd_df)

        effects = self._compute_effects(
            diff_matrix,
            self._params,
            self._vcv,
            is_cates=True,
            include_inference=return_results_dict,
        )

        if return_results_dict:
            results = {"outcome": self.Y}
            results.update(effects)
            return results
        return effects["cate"]

    def predict(
        self,
        df: PandasConvertibleDataFrame,
        *,
        return_results_dict: bool = False,
        mode: str = "cate",
    ) -> np.ndarray | dict:
        """Generate predicted conditional average treatment effects (CATEs) or outcomes.

        When mode is "outcome", the function returns predicted outcomes.

        When mode is "cate", the function returns predicted CATEs, behaving as an alias for `estimate_cate`.

        Parameters
        ----------
        df : PandasConvertibleDataFrame
            Dataframe containing the data to estimate CATEs for. Supported formats:
                pandas DataFrame, PySpark DataFrame, Polars DataFrame, or Any object with `toPandas()` or `to_pandas()` method
        return_results_dict : bool
            If True, the function returns a dictionary containing CATEs, standard errors, t-statistics, and p-values.
            If False, the function returns a numpy array containing CATEs alone.
            Does not have any effect when mode is "outcome".
        mode : str
            The mode of prediction. Supported modes are "cate" and "outcome".
            If "cate", the function returns CATEs.
            If "outcome", the function returns predicted outcomes.

        Returns
        -------
        np.ndarray | dict
            CATEs or dictionary containing CATEs, standard errors, t-statistics, and p-values.

        Examples
        --------
        ```{python}
        cates = ilr.predict(df)
        cates[:5]
        ```
        ```{python}
        res = ilr.predict(df, return_results_dict=True)
        res.keys()
        ```
        """
        if mode == "cate":
            return self.estimate_cate(df, return_results_dict=return_results_dict)
        elif mode == "outcome":
            pd_df = self._convert_dataframe_to_pandas(df, self.G)
            _, X = self._create_design_matrix(pd_df)
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
        ilr.prettify_treatment_effects()
        ```
        ```{python}
        ## Using a custom GATE
        custom_gate = ilr.estimate_ate(df_filtered, return_results_dict=True, group="My Custom Group")
        ilr.prettify_treatment_effects(custom_gate)
        ```
        """
        if effects is None:
            effects_to_prettify = self._treatment_effects
        else:
            effects_to_prettify = effects

        n_outcomes = len(self.Y)

        final_results = {}

        for i, k in enumerate(effects_to_prettify.keys()):
            try:
                group = k.split("-")[0]
                membership = k.split("-")[1]
            except IndexError:
                group = k
                membership = None
            if i == 0:
                final_results["group"] = [group] * n_outcomes
                final_results["membership"] = [membership] * n_outcomes
                for stat, value in effects_to_prettify[k].items():
                    if isinstance(value, list):
                        final_results[stat] = value.copy()
                    elif isinstance(value, np.ndarray):
                        final_results[stat] = value.flatten().copy()
                    elif isinstance(value, int):
                        final_results[stat] = [value] * n_outcomes
            else:
                final_results["group"] += [group] * n_outcomes
                final_results["membership"] += [membership] * n_outcomes
                for stat, value in effects_to_prettify[k].items():
                    if isinstance(value, list):
                        final_results[stat] += value
                    elif isinstance(value, np.ndarray):
                        final_results[stat] = np.hstack(
                            [final_results[stat], value.flatten()]
                        )
                    elif isinstance(value, int):
                        final_results[stat] += [value] * n_outcomes

        return pd.DataFrame(final_results)

    @timer("Model Fitting")
    def _fit(self, X: np.ndarray, y: np.ndarray, cov_type: str = "nonrobust"):
        INFO("Fitting regression model...")

        def fit(X, y):
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            fitted_values = X @ params
            resid = y - fitted_values
            n = X.shape[0]
            k = X.shape[1]
            XtX_inv = np.linalg.pinv(X.T @ X)
            if cov_type in ("HC0", "HC1"):
                E = resid**2
                if cov_type == "HC1":
                    E *= n / (n - k)
                XEX = np.einsum("ni,nj,no->oij", X, X, E)
                vcv = XtX_inv @ XEX @ XtX_inv
            else:
                rss = np.sum(resid**2, axis=0)
                sigma_squared_hat = rss / (n - k)
                XtX_inv = np.linalg.pinv(X.T @ X)
                vcv = np.einsum("o,ij->oij", sigma_squared_hat, XtX_inv)
            return params, vcv, fitted_values, resid

        params, vcv, fitted_values, residuals = fit(X, y)

        self._params = params
        self._vcv = vcv
        self._std_err = np.sqrt(np.diagonal(vcv, axis1=1, axis2=2)).T
        self._fitted_values = fitted_values
        self._residuals = residuals
        self._treatment_effects = {}

    @timer("Design Matrix Creation")
    def _create_design_matrix(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray] | NoReturn:
        try:
            DEBUG("Creating model design matrix...")
            y, X = patsy.dmatrices(self.formula, data=df, NA_action="raise")  # pyright: ignore[reportAttributeAccessIssue]

            self._X_design_info = X.design_info

            y = np.array(y)
            X = np.array(X)

            return y, X
        except patsy.PatsyError as e:
            if "factor contains missing values" in str(e):
                raise ValueError(
                    "Input DataFrame contains missing values. Please handle missing values before proceeding."
                )
            else:
                raise e
        except Exception as e:
            raise e

    @timer("Difference Matrix Creation")
    def _create_difference_matrix(self, df: pd.DataFrame) -> np.ndarray | NoReturn:
        try:
            DEBUG("Creating treatment difference matrix...")
            original_t = df[self.T].copy()
            if self._X_design_info is None:
                y, X = patsy.dmatrices(self.formula, data=df, NA_action="raise")  # pyright: ignore[reportAttributeAccessIssue]
                self._X_design_info = X.design_info

            if self._discrete_treatment:
                df[self.T] = 0
                X0 = patsy.dmatrix(self._X_design_info, data=df, NA_action="raise")  # pyright: ignore[reportAttributeAccessIssue]
                df[self.T] = 1
                X1 = patsy.dmatrix(self._X_design_info, data=df, NA_action="raise")  # pyright: ignore[reportAttributeAccessIssue]
            else:
                X0 = patsy.dmatrix(self._X_design_info, data=df, NA_action="raise")  # pyright: ignore[reportAttributeAccessIssue]
                df[self.T] = df[self.T] + 1
                X1 = patsy.dmatrix(self._X_design_info, data=df, NA_action="raise")  # pyright: ignore[reportAttributeAccessIssue]

            df[self.T] = original_t

            X1 = np.array(X1)
            X0 = np.array(X0)

            diff = X1 - X0

            return diff
        except patsy.PatsyError as e:
            if "factor contains missing values" in str(e):
                raise ValueError(
                    "Input DataFrame contains missing values. Please handle missing values before proceeding."
                )
            else:
                raise e
        except Exception as e:
            raise e

    @staticmethod
    def _compute_effects(
        diff_matrix: np.ndarray,
        params: np.ndarray,
        vcv: np.ndarray,
        n_treated: int | None = None,
        is_cates: bool = False,
        include_inference: bool = True,
    ) -> dict:
        if is_cates:
            d = diff_matrix
        else:
            d = np.mean(diff_matrix, axis=0).reshape(1, -1)

        effect = d @ params
        n = diff_matrix.shape[0]
        if include_inference:
            std_err = np.sqrt(np.einsum("nj,ojk,nk->no", d, vcv, d))
            t_stat = np.where(std_err > 0, effect / std_err, 0)
            pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - params.shape[0]))
        else:
            std_err, t_stat, pval = None, None, None

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
        _diff_matrix: np.ndarray,
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
                mask = np.array(df[group] == membership)
                treated_mask = (
                    np.array(df[df[group] == membership][self.T] == 1)
                    if self._discrete_treatment
                    else None
                )
                group_key = f"{group}-{membership}"
                group_info.append((group_key, mask, treated_mask))

        def process_group(group_key, mask, treated_mask):
            diff_matrix_filtered = _diff_matrix[mask]
            n_treated = int(treated_mask.sum()) if self._discrete_treatment else None
            effects = self._compute_effects(
                diff_matrix=diff_matrix_filtered,
                params=self._params,
                vcv=self._vcv,
                n_treated=n_treated,
            )
            return group_key, effects

        DEBUG(f"Starting parallel processing with {n_jobs} jobs")
        results: Any = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_group)(group_key, mask, treated_mask)
            for group_key, mask, treated_mask in group_info
        )

        for group_key, effects in results:
            self._treatment_effects[group_key] = {"outcome": self.Y}
            self._treatment_effects[group_key].update(effects)

    @staticmethod
    def _create_formula(
        Y: list[str],
        T: str,
        G: list[str],
        X: list[str],
        W: list[str],
        discrete_treatment: bool = False,
        xformula: str | None = None,
    ) -> str:
        formula = " + ".join([f"Q('{y}')" for y in Y])

        if discrete_treatment:
            treatment = f"C(Q('{T}'))"
        else:
            treatment = f"Q('{T}')"

        formula += f" ~ {treatment}"

        for g in G:
            formula += f" + C(Q('{g}'))*{treatment}"

        for x in X:
            formula += f" + Q('{x}')*{treatment}"

        for w in W:
            formula += f" + Q('{w}')"

        if xformula:
            formula += f" {xformula}"

        return formula

    def __str__(self):
        """
        Returns a string representation of the InteractiveLinearRegression object.

        Returns
        -------
        str
            A string containing information about the InteractiveLinearRegression object.
        """
        summary = (
            "================== InteractiveLinearRegression Object ==================\n"
            + f"Outcome Variable: {self.Y}\n"
            + f"Treatment Variable: {self.T}\n"
            + f"Discrete Treatment: {self._discrete_treatment}\n"
            + f"Group Variables: {self.G}\n"
            + f"Features/Confounders for Heterogeneity (X): {self.X}\n"
            + f"Features/Confounders as Controls (W): {self.W}\n"
            + f"Formula: {self.formula}\n"
        )

        return summary

    def __getstate__(self):
        """Fix to remove non-serializable patsy objects."""
        state = self.__dict__.copy()
        if "_X_design_info" in state:
            state["_X_design_info"] = None
        return state
