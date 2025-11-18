from typing import NoReturn

import numpy as np
import pandas as pd
import patsy

from caml._generics.decorators import timer
from caml._generics.logging import DEBUG, INFO


class OLSMixin:
    """Mixin class for Ordinary Least Squares (OLS) regression."""

    # TODO: Break inference apart
    @staticmethod
    @timer("OLS Estimation")
    def _fit_ols(
        X: np.ndarray, y: np.ndarray, cov_type: str = "nonrobust"
    ) -> dict[str, np.ndarray]:
        INFO("Fitting regression model...")

        if cov_type not in ("nonrobust", "HC0", "HC1"):
            raise ValueError("cov_type must be 'nonrobust', 'HC0', or 'HC1'")

        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted_values = X @ params
        residuals = y - fitted_values
        n = X.shape[0]
        k = X.shape[1]
        XtX_inv = np.linalg.pinv(X.T @ X)
        if cov_type in ("HC0", "HC1"):
            E = residuals**2
            if cov_type == "HC1":
                E *= n / (n - k)
            XEX = np.einsum("ni,nj,no->oij", X, X, E)
            vcv = XtX_inv @ XEX @ XtX_inv
        else:
            rss = np.sum(residuals**2, axis=0)
            sigma_squared_hat = rss / (n - k)
            XtX_inv = np.linalg.pinv(X.T @ X)
            vcv = np.einsum("o,ij->oij", sigma_squared_hat, XtX_inv)

        std_err = np.sqrt(np.diagonal(vcv, axis1=1, axis2=2)).T

        return {
            "params": params,
            "vcv": vcv,
            "std_err": std_err,
            "fitted_values": fitted_values,
            "residuals": residuals,
        }

    @staticmethod
    @timer("Design Matrix Creation")
    def _create_design_matrix(
        df: pd.DataFrame,
        formula: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | NoReturn:
        try:
            DEBUG("Creating model design matrix...")
            y, X = patsy.dmatrices(formula, data=df, NA_action="raise")  # pyright: ignore[reportAttributeAccessIssue]

            design_info = X.design_info
            y = np.array(y)
            X = np.array(X)

            return y, X, design_info
        except patsy.PatsyError as e:
            if "factor contains missing values" in str(e):
                raise ValueError(
                    "Input DataFrame contains missing values. Please handle missing values before proceeding."
                )
            else:
                raise e
        except Exception as e:
            raise e
