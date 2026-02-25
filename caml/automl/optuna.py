"""Optuna-based backend for CATE model selection.

Wraps `optuna` to perform Bayesian hyperparameter optimization (`~~optuna.samplers.TPESampler` by default)
over candidate CATE estimators. Study state is persisted to a local SQLite database
under `~/.caml/studies/`, by default, enabling [`optuna-dashboard`](https://optuna-dashboard.readthedocs.io/en/latest/#) integration.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import optuna
from sklearn.base import BaseEstimator

from caml.data.dataset import CausalDataset
from caml.samplers.splitters import create_splitter

from .base_backend import BaseTunerBackend
from .search_space import NuisanceModelSpec

if TYPE_CHECKING:
    from caml.scorers.base_scorer import CateScorer


class OptunaBackend(BaseTunerBackend):
    """Optuna backend for CATE model selection.

    Parameters
    ----------
    direction
        Optimization direction — `"minimize"` (default) or `"maximize"`.
    sampler
        Optuna sampler instance. Defaults to `~~optuna.samplers.TPESampler`.
    storage
        Optuna-compatible storage URL. When `None`, a SQLite database is
        created at `~/.caml/studies/<study_name>.db`.
    study_name
        Human-readable study name. Auto-generated if `None`.
    load_if_exists
        If `True` and `study_name` is provided, resumes an existing study
        with the same name instead of creating a new one.
    **kwargs
        Additional kwargs forwarded to `optuna.create_study()`.

    Notes
    -----
    The default sampler is `~~optuna.samplers.TPESampler` (Tree-structured Parzen Estimator),
    which works well for mixed categorical/continuous search spaces typical of
    CATE estimator selection.

    Study results are viewable via [`optuna-dashboard`](https://optuna-dashboard.readthedocs.io/en/latest/#):

    ```bash
    optuna-dashboard sqlite:///~/.caml/studies/<study_name>.db
    ```

    See Also
    --------
    `~~base_backend.BaseTunerBackend` : ABC this class extends.

    `~~base_backend.TunerBackend` : Protocol this class satisfies.

    `~~auto_cate.AutoCATE` : High-level interface that uses this backend.

    Examples
    --------
    ```{python}
    from caml.automl import OptunaBackend, TunerBackend

    backend = OptunaBackend(direction="minimize")
    print(f"Study: {backend.study_name}")
    print(f"Storage: {backend.storage}")

    # Protocol conformance
    assert isinstance(backend, TunerBackend)
    ```
    """

    def __init__(
        self,
        direction: str = "minimize",
        sampler: optuna.samplers.BaseSampler | None = None,
        storage: str | optuna.storages.BaseStorage | None = None,
        study_name: str | None = None,
        load_if_exists: bool = True,
        **kwargs,
    ):
        self.direction = direction
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.study_name = study_name or f"autocate-{uuid.uuid4().hex[:8]}"
        self.storage = self._resolve_storage(storage, self.study_name)

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=load_if_exists if study_name is not None else False,
            **kwargs,
        )

    @staticmethod
    def _resolve_storage(storage, study_name):
        """Resolve storage URL, defaulting to a local SQLite database.

        Parameters
        ----------
        storage
            Explicit storage URL or `None`.
        study_name
            Study name used to derive the default SQLite path.

        Returns
        -------
        str
            Optuna-compatible storage URL.
        """
        if storage is not None:
            return storage

        base = Path.home() / ".caml" / "studies"
        base.mkdir(parents=True, exist_ok=True)

        return f"sqlite:///{base / (study_name + '.db')}"

    def optimize(
        self,
        objective: Callable,
        n_trials: int,
        n_jobs: int = 1,
        timeout: int | None = None,
        **kwargs,
    ) -> optuna.study.Study:
        """Run the Optuna optimization loop.

        Parameters
        ----------
        objective
            Callable created by `~~optuna.OptunaBackend.create_objective()`.
        n_trials
            Number of Optuna trials to run.
        n_jobs
            Number of parallel workers (`1` = sequential).
        timeout
            Time limit in seconds. `None` means no limit.
        **kwargs
            Additional kwargs forwarded to `optuna.study.Study.optimize()`.

        Returns
        -------
        optuna.study.Study
            Completed Optuna study containing trial results.
        """
        self.study.optimize(
            objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout, **kwargs
        )

        return self.study

    def create_objective(
        self,
        data: CausalDataset,
        scorer: CateScorer,
        candidate_cate_estimators: list[str],
        cv: int,
        outcome_model: BaseEstimator | None,
        treatment_model: BaseEstimator | None,
        regression_model: BaseEstimator | None,
    ):
        """Build the Optuna objective function for CATE model selection.

        The returned callable is passed to `~~optuna.OptunaBackend.optimize()`. Each trial selects a
        candidate estimator, samples its hyperparameters, and evaluates it via
        cross-validated scoring.

        Parameters
        ----------
        data
            Training data for cross-validated evaluation.
        scorer
            CATE scorer instance (e.g., `~~r_loss.RLoss`, `~~dr_loss.DRLoss`).
        candidate_cate_estimators
            Registry keys of estimators to search over.
        cv
            Number of cross-validation folds.
        outcome_model
            Pre-tuned outcome nuisance model, or `None`.
        treatment_model
            Pre-tuned treatment nuisance model, or `None`.
        regression_model
            Pre-tuned regression nuisance model, or `None`.

        Returns
        -------
        callable
            Objective function with signature `(optuna.Trial) -> float`.

        Notes
        -----
        Inside each trial the objective:

        1. Samples a candidate estimator via `trial.suggest_categorical()`.
        2. Samples estimator-specific hyperparameters from its `default_search_space`.
        3. Injects pre-tuned nuisance models where required.
        4. Scores via `cv`-fold cross-validation, returning the mean score.
        """
        from caml.registry.registry import AVAILABLE_CATE_ESTIMATORS

        def objective(trial: optuna.Trial) -> float:
            estimator_name = trial.suggest_categorical(
                "estimator", candidate_cate_estimators
            )
            estimator = AVAILABLE_CATE_ESTIMATORS[estimator_name]["estimator"]()
            for param in estimator.default_search_space:
                if isinstance(param, NuisanceModelSpec):
                    if param.model_type == "regression":
                        estimator.set_params(**{param.name: regression_model})
                    elif param.model_type == "outcome":
                        estimator.set_params(**{param.name: outcome_model})
                    else:
                        estimator.set_params(**{param.name: treatment_model})
                else:
                    estimator.set_params(
                        **{
                            param.name: param.to_optuna(
                                trial, prefix=f"{estimator_name}__"
                            )
                        }
                    )

            cv_splitter = create_splitter(
                cv=cv,
                random_state=trial.number,
                stratified=data.treatment_type.is_discrete(),
            )
            scores = []
            for train_idx, test_idx in cv_splitter.split(data.X, data.T.ravel()):
                estimator.fit(data.sample(train_idx))
                scores.append(scorer(estimator, data.sample(test_idx)))
            return float(np.mean(scores))

        return objective
