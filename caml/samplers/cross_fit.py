import numpy as np
import sklearn
from sklearn.model_selection import cross_val_predict

from caml._generics.utils import arr_at_least_2d
from caml.data import CausalDataset
from caml.samplers.splitters import create_splitter

sklearn.set_config(enable_metadata_routing=True)


class CrossFitter:
    """Cross-fitting engine for orthogonal scores."""

    def __init__(self, cv: int = 3, random_state: int | None = None):
        self.cv = cv
        self.random_state = random_state

    def fit_predict_nuisances(
        self, data: CausalDataset, outcome_model, treatment_model
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit nuisance models with cross-fitting and return out-of-fold predictions.

        Returns
        -------
        m_hat : np.ndarray
            Out-of-fold outcome predictions E[Y|X,W]
        e_hat : np.ndarray
            Out-of-fold treatment predictions E[T|X,W]
        """
        # Prepare features
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

        if data.outcome_type.is_discrete():
            m_hat: np.ndarray = m_hat[:, 1]
        if data.treatment_type.is_discrete():
            e_hat: np.ndarray = e_hat[:, 1]

        return arr_at_least_2d(m_hat), arr_at_least_2d(e_hat)

    def fit_predict_nuisances_dr(
        self, data: CausalDataset, outcome_model, treatment_model
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit nuisance models for DR-loss (need mu_0, mu_1, e).

        Returns
        -------
        mu_0 : np.ndarray
            E[Y|X,W,T=0]
        mu_1 : np.ndarray
            E[Y|X,W,T=1]
        e_hat : np.ndarray
            E[T|X,W]
        """
        # Prepare features
        XW = np.hstack([data.X, data.W]) if data.W is not None else data.X

        # Create splitter
        outcome_splitter = create_splitter(
            cv=self.cv,
            random_state=self.random_state,
            stratified=data.outcome_type.is_discrete(),
            groups=None,  # Groups not supported yet
        )
        mu_0 = cross_val_predict(  # pyright: ignore[reportAssignmentType]
            outcome_model,
            XW,
            data.Y.ravel(),
            cv=outcome_splitter,
            method="predict_proba" if data.outcome_type.is_discrete() else "predict",
        )
        mu_1 = cross_val_predict(  # pyright: ignore[reportAssignmentType]
            outcome_model,
            XW,
            data.Y.ravel(),
            cv=outcome_splitter,
            method="predict_proba" if data.outcome_type.is_discrete() else "predict",
        )

        treatment_splitter = create_splitter(
            cv=self.cv,
            random_state=self.random_state,
            stratified=True,
            groups=None,  # Groups not supported yet
        )
        e_hat = cross_val_predict(  # pyright: ignore[reportAssignmentType]
            treatment_model,
            XW,
            data.T.ravel(),
            cv=treatment_splitter,
            method="predict_proba",
        )

        if data.outcome_type.is_discrete():
            mu_0: np.ndarray = mu_0[:, 1]
            mu_1: np.ndarray = mu_1[:, 1]

        e_hat: np.ndarray = e_hat[:, 1]

        return arr_at_least_2d(mu_0), arr_at_least_2d(mu_1), arr_at_least_2d(e_hat)
