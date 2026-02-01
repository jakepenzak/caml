"""Cross-fitting engine for orthogonal scores and nuisance model estimation.

Provides the core functionality for generating out-of-fold predictions to be leveraged in scoring.
"""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict

from caml._generics.utils import arr_at_least_2d
from caml.data import CausalDataset

from .splitters import create_splitter


class CrossFitter:
    """Cross-fitting engine for orthogonal scores.

    This class provides methods to fit and predict nuisance models using cross-fitting, and is leveraged in
    orthogonal scoring functions such as RLoss, DRLoss, etc. This class can be used for general cross-fitting
    of outcome and treatment models as well.

    Parameters
    ----------
    cv
        Number of cross-fitting folds.
    random_state
        Random state for cross-fitting folds.

    Examples
    --------
    ```{python}
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.samplers import CrossFitter

    # Generate synthetic data
    gen = SyntheticDataGenerator(
        n_cont_modifiers=3,
        n_obs=500,
        seed=42
    )

    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )

    # Initialize cross-fitter
    cross_fitter = CrossFitter(cv=5)

    # Define nuisance models
    outcome_model = LinearRegression()
    treatment_model = LogisticRegression()

    # Fit and predict nuisances using cross-fitting
    m_hat, e_hat = cross_fitter.fit_predict_nuisances_dml(
        data=data,
        outcome_model=outcome_model,
        treatment_model=treatment_model
    )

    print("Outcome predictions shape:", m_hat.shape)
    print("Treatment predictions shape:", e_hat.shape)
    ```
    """

    def __init__(self, cv: int = 3, random_state: int | None = None):
        self.cv = cv
        self.random_state = random_state

    def fit_predict_outcome_model(
        self, data: CausalDataset, outcome_model: BaseEstimator
    ) -> np.ndarray:
        r"""Cross-fit the outcome model.

        Estimates the outcome model $\mathbb{E}[Y \mid X,W]$ using cross-fitting and returns out-of-fold predictions.

        Binary, continuous, and categorical outcomes are supported.

        Parameters
        ----------
        data
            Causal dataset containing features, treatment, outcome, and covariates.
        outcome_model
            A scikit-learn compatible estimator for the outcome model.

        Returns
        -------
        np.ndarray
            Out-of-fold predictions of the outcome model $\mathbb{E}[Y \mid X,W]$.
        """
        XW = np.hstack([data.X, data.W]) if data.W is not None else data.X

        outcome_splitter = create_splitter(
            cv=self.cv,
            random_state=self.random_state,
            stratified=data.outcome_type.is_discrete(),
            groups=None,  # Groups not supported yet
        )
        m_hat = cross_val_predict(  # pyright: ignore[reportAssignmentType]
            outcome_model,
            XW,
            data.Y.ravel(),
            cv=outcome_splitter,
            method="predict_proba" if data.outcome_type.is_discrete() else "predict",
        )

        if data.outcome_type.is_discrete():
            m_hat: np.ndarray = m_hat[:, 1]

        return arr_at_least_2d(m_hat)

    def fit_predict_treatment_model(
        self, data: CausalDataset, treatment_model: BaseEstimator
    ) -> np.ndarray:
        r"""Cross-fit the treatment model.

        Estimates the treatment model $\mathbb{E}[T \mid X,W]$ using cross-fitting and returns out-of-fold predictions.

        Binary, continuous, and categorical treatments are supported.

        Parameters
        ----------
        data
            Causal dataset containing features, treatment, outcome, and covariates.
        treatment_model
            A scikit-learn compatible estimator for the treatment model.

        Returns
        -------
        np.ndarray
            Out-of-fold predictions of the treatment model $\mathbb{E}[T \mid X,W]$.
        """
        XW = np.hstack([data.X, data.W]) if data.W is not None else data.X

        treatment_splitter = create_splitter(
            cv=self.cv,
            random_state=self.random_state,
            stratified=data.treatment_type.is_discrete(),
            groups=None,  # Groups not supported yet
        )
        e_hat = cross_val_predict(  # pyright: ignore[reportAssignmentType]
            treatment_model,
            XW,
            data.T.ravel(),
            cv=treatment_splitter,
            method="predict_proba" if data.treatment_type.is_discrete() else "predict",
        )

        if data.treatment_type.is_discrete():
            e_hat: np.ndarray = e_hat[:, 1]

        return arr_at_least_2d(e_hat)

    def fit_predict_regression_model(
        self,
        data: CausalDataset,
        regression_model: BaseEstimator,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Cross-fit the regression model for treatment and control groups seperately.

        Estimates the regression models for treatment, $\mathbb{E}[Y \mid X,W,T=1]$, and control, $\mathbb{E}[Y \mid X,W,T=0]$, groups separately
        using cross-fitting, and returns out-of-fold predictions for all observations.

        Binary, continuous, and categorical outcomes are supported.

        Parameters
        ----------
        data
            Causal dataset containing features, treatment, outcome, and covariates.
        regression_model
            A scikit-learn compatible estimator for the regression model.

        Returns
        -------
        mu_0 : np.ndarray
            Out-of-fold predictions based on control group model $\mathbb{E}[Y \mid X,W,T=0]$.
        mu_1 : np.ndarray
            Out-of-fold predictions based on treatment group $\mathbb{E}[Y \mid X,W,T=1]$.
        """
        XW = np.hstack([data.X, data.W]) if data.W is not None else data.X
        T_flat = data.T.ravel()
        Y_flat = data.Y.ravel()

        outcome_splitter = create_splitter(
            cv=self.cv,
            random_state=self.random_state,
            stratified=data.outcome_type.is_discrete(),
            groups=None,  # Groups not supported yet
        )

        # Initialize arrays for all predictions
        mu_0 = np.zeros(len(XW))
        mu_1 = np.zeros(len(XW))

        for train_idx, test_idx in outcome_splitter.split(XW, Y_flat):
            # For mu_0: train only on control group (T=0) within training fold
            control_mask = T_flat[train_idx] == 0
            train_control_idx = train_idx[control_mask]

            if len(train_control_idx) > 0:
                model_0 = clone(regression_model)
                model_0.fit(XW[train_control_idx], Y_flat[train_control_idx])  # pyright: ignore[reportAttributeAccessIssue]

                # Predict on test fold (all observations)
                if data.outcome_type.is_discrete():
                    mu_0[test_idx] = model_0.predict_proba(XW[test_idx])[:, 1]  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    mu_0[test_idx] = model_0.predict(XW[test_idx])  # pyright: ignore[reportAttributeAccessIssue]

            # For mu_1: train only on treatment group (T=1) within training fold
            treatment_mask = T_flat[train_idx] == 1
            train_treatment_idx = train_idx[treatment_mask]

            if len(train_treatment_idx) > 0:
                model_1 = clone(regression_model)
                model_1.fit(XW[train_treatment_idx], Y_flat[train_treatment_idx])  # pyright: ignore[reportAttributeAccessIssue]

                # Predict on test fold (all observations)
                if data.outcome_type.is_discrete():
                    mu_1[test_idx] = model_1.predict_proba(XW[test_idx])[:, 1]  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    mu_1[test_idx] = model_1.predict(XW[test_idx])  # pyright: ignore[reportAttributeAccessIssue]

        return arr_at_least_2d(mu_0), arr_at_least_2d(mu_1)

    def fit_predict_nuisances_dml(
        self,
        data: CausalDataset,
        outcome_model: BaseEstimator,
        treatment_model: BaseEstimator,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Cross-fit nuisance models for partially linear model (PLM) formulations of DML.

        This method fits the outcome model $\mathbb{E}[Y \mid X,W]$ and treatment model $\mathbb{E}[T \mid X,W]$ using cross-fitting.

        Parameters
        ----------
        data
            Causal dataset containing features, treatment, outcome, and covariates.
        outcome_model
            A scikit-learn compatible estimator for the outcome model.
        treatment_model
            A scikit-learn compatible estimator for the treatment model.

        Returns
        -------
        m_hat : np.ndarray
            Out-of-fold outcome predictions $\mathbb{E}[Y \mid X,W]$
        l_hat : np.ndarray
            Out-of-fold treatment predictions $\mathbb{E}[T \mid X,W]$
        """
        m_hat = self.fit_predict_outcome_model(data, outcome_model)
        l_hat = self.fit_predict_treatment_model(data, treatment_model)

        return m_hat, l_hat

    def fit_predict_nuisances_dr(
        self,
        data: CausalDataset,
        regression_model: BaseEstimator,
        treatment_model: BaseEstimator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Fit nuisance models for interactive regression model (IRM) formulation of DML (Doubly Robust).

        This method fits the regression models for treatment and control groups separately,
        $\mathbb{E}[Y \mid X,W,T=1]$ and $\mathbb{E}[Y \mid X,W,T=0]$, as well as the treatment model $\mathbb{E}[T \mid X,W]$ using cross-fitting.

        Parameters
        ----------
        data
            Causal dataset containing features, treatment, outcome, and covariates.
        regression_model
            A scikit-learn compatible estimator for the regression model.
        treatment_model
            A scikit-learn compatible estimator for the treatment model.

        Returns
        -------
        mu_0 : np.ndarray
            Out-of-fold predictions $\mathbb{E}[Y \mid X,W,T=0]$
        mu_1 : np.ndarray
            Out-of-fold predictions $\mathbb{E}[Y \mid X,W,T=1]$
        e_hat : np.ndarray
            Out-of-fold treatment predictions $\mathbb{E}[T \mid X,W]$
        """
        mu_0, mu_1 = self.fit_predict_regression_model(data, regression_model)
        e_hat = self.fit_predict_treatment_model(data, treatment_model)

        return mu_0, mu_1, e_hat
