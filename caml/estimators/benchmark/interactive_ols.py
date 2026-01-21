from typing import Any, NoReturn, Sequence

import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
from joblib import Parallel, delayed

from caml._base.abstract import BaseCamlEstimator
from caml._base.mixins import OLSMixin
from caml._generics.decorators import experimental, timer
from caml._generics.interfaces import FittedAttr, PandasConvertibleDataFrame
from caml._generics.logging import DEBUG, INFO


@experimental
class InteractiveLinearRegression(BaseCamlEstimator, OLSMixin):
    r"""Interactive linear regression estimator with explicit treatment interaction terms, enabling precision improvements & heterogeneous treatment effect discovery.

    **`InteractiveLinearRegression` is experimental and may change significantly in future versions.**

    This class estimates a standard linear regression model, with treatment-covariate interaction terms, for any number of continuous or binary outcomes and a single continuous or binary treatment,
    and provides estimates for the Average Treatment Effects (ATEs) and Group Average Treatment Effects (GATEs) out of the box. Additionally,
    methods are provided for estimating custom GATEs & Conditional Average Treatment Effects (CATEs) of individual observations, which can also be used for out-of-sample predictions.
    Note, this method assumes linear treatment effects and heterogeneity, which is typically sufficient when primarily concerned with ATEs and GATEs.

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
    from caml.cross_section import InteractiveLinearRegression
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
        pd_df = self._convert_dataframe_to_pandas(df, self.G)
        if self._discrete_treatment:
            if len(pd_df[self.T].unique()) != 2:
                raise ValueError("Treatment variable must be binary")
        y, X, self._X_design_info = self._create_design_matrix(pd_df, self.formula)
        res = self._fit_ols(X, y, cov_type=cov_type)
        self._params, self._vcv, self._std_err, self._fitted_values, self._residuals = (
            res.values()
        )
        self._treatment_effects = {}
        self._fitted = True
        if estimate_effects:
            diff_matrix = self._create_difference_matrix(pd_df)
            self._treatment_effects = self.estimate(  # pyright: ignore[reportAttributeAccessIssue]
                pd_df,
                estimand="ate",
                query=None,
                return_results_dict=True,
                _diff_matrix=diff_matrix,
            )
            self._estimate_gates_parallel(
                pd_df, _diff_matrix=diff_matrix, n_jobs=n_jobs
            )

    def estimate(
        self,
        df: PandasConvertibleDataFrame,
        *,
        estimand: str,
        query: str | None = None,
        return_results_dict: bool = False,
        _diff_matrix: np.ndarray | None = None,
    ) -> Any:
        """ """
        all_kwargs = {
            "return_results_dict": return_results_dict,
            "_diff_matrix": _diff_matrix,
        }

        if not self._fitted:
            raise RuntimeError("Model must be fitted before estimating ATEs.")

        return super().estimate(df, estimand=estimand, query=query, **all_kwargs)

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
            return self.estimate(
                df, estimand="cate", query=None, return_results_dict=return_results_dict
            )
        elif mode == "outcome":
            pd_df = self._convert_dataframe_to_pandas(df, self.G)
            _, X, _ = self._create_design_matrix(pd_df, self._formula)
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
        custom_gate = ilr.estimate(df, estimand="gate", query="X3_binary == 0 & X1_continuous < 5", return_results_dict=True)
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
                estimand = k.split("--")[0]
                group = k.split("--")[1]
            except IndexError:
                estimand = k
                group = None
            if i == 0:
                final_results["estimand"] = [estimand] * n_outcomes
                final_results["group"] = [group] * n_outcomes
                for stat, value in effects_to_prettify[k].items():
                    if isinstance(value, list):
                        final_results[stat] = value.copy()
                    elif isinstance(value, np.ndarray):
                        final_results[stat] = value.flatten().copy()
                    elif isinstance(value, int):
                        final_results[stat] = [value] * n_outcomes
            else:
                final_results["estimand"] += [estimand] * n_outcomes
                final_results["group"] += [group] * n_outcomes
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

    @timer("ATE Estimation")
    def _estimate_ate(
        self,
        df: pd.DataFrame,
        return_results_dict: bool,
        _diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray | dict:
        INFO("Estimating Average Treatment Effect (ATE)...")

        return self._estimate_effect_common(
            df,
            return_results_dict=return_results_dict,
            estimand="ATE",
            group_name="Overall",
            _diff_matrix=_diff_matrix,
        )

    @timer("GATE Estimation")
    def _estimate_gate(
        self,
        df: pd.DataFrame,
        query: str,
        return_results_dict: bool,
        _diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray | dict:
        INFO("Estimating Group Average Treatment Effect (GATE)...")

        df_filtered = df.query(query)

        return self._estimate_effect_common(
            df_filtered,
            return_results_dict=return_results_dict,
            estimand="GATE",
            group_name=query,
            _diff_matrix=_diff_matrix,
        )

    @timer("ATT Estimation")
    def _estimate_att(
        self,
        df: pd.DataFrame,
        return_results_dict: bool,
        _diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray | dict:
        INFO("Estimating Average Treatment Effect on the Treated (ATT)...")

        if self._discrete_treatment:
            df_filtered = df.query(f"{self.T} == 1")
        else:
            raise ValueError(
                "ATT estimation is not supported for continuous treatments."
            )

        return self._estimate_effect_common(
            df_filtered,
            return_results_dict=return_results_dict,
            estimand="ATT",
            group_name="Treated",
            _diff_matrix=_diff_matrix,
        )

    @timer("ATC Estimation")
    def _estimate_atc(
        self,
        df: pd.DataFrame,
        return_results_dict: bool,
        _diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray | dict:
        INFO("Estimating Average Treatment Effect on the Control (ATC)...")

        if self._discrete_treatment:
            df_filtered = df.query(f"{self.T} == 0")
        else:
            raise ValueError(
                "ATC estimation is not supported for continuous treatments."
            )

        return self._estimate_effect_common(
            df_filtered,
            return_results_dict=return_results_dict,
            estimand="ATC",
            group_name="Control",
            _diff_matrix=_diff_matrix,
        )

    @timer("CATE Estimation")
    def _estimate_cate(
        self,
        df: pd.DataFrame,
        return_results_dict: bool,
        _diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray | dict:
        INFO("Estimating Conditional Average Treatment Effects (CATEs)...")
        return self._estimate_effect_common(
            df,
            return_results_dict=return_results_dict,
            estimand="cate",
            group_name="",
            _diff_matrix=_diff_matrix,
        )

    def _estimate_effect_common(
        self,
        df: pd.DataFrame,
        return_results_dict: bool,
        estimand: str,
        group_name: str,
        _diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray | dict:
        if _diff_matrix is None:
            diff_matrix = self._create_difference_matrix(df)
        else:
            diff_matrix = _diff_matrix

        n_treated = int(df[self.T].sum()) if self._discrete_treatment else None

        effects = self._compute_effects(
            diff_matrix=diff_matrix,
            params=self._params,
            vcv=self._vcv,
            n_treated=n_treated,
            is_cates=True if estimand == "cate" else False,
            include_inference=return_results_dict,
        )

        if return_results_dict:
            results = {}
            key = f"{estimand}--{group_name}"
            results[key] = {"outcome": self.Y}
            results[key].update(effects)
            return results

        return effects["effect"]

    @timer("Difference Matrix Creation")
    def _create_difference_matrix(self, df: pd.DataFrame) -> np.ndarray | NoReturn:
        try:
            DEBUG("Creating treatment difference matrix...")
            original_t = df[self.T].copy()
            if self._X_design_info is None:
                _, _, self._X_design_info = self._create_design_matrix(
                    df, formula=self._formula
                )

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

        results = {
            "effect": effect,
            "std_err": std_err,
            "t_stat": t_stat,
            "pval": pval,
        }
        if not is_cates:
            results["n"] = n
            results["n_treated"] = n_treated
            results["n_control"] = n_control

        return results

    @timer("Prespecified GATE Estimation")
    def _estimate_gates_parallel(
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
                group_key = f"{group}={membership}"
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
            self._treatment_effects[f"GATE--{group_key}"] = {"outcome": self.Y}
            self._treatment_effects[f"GATE--{group_key}"].update(effects)

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
