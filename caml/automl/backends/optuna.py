"""Optuna backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import optuna
from sklearn.base import BaseEstimator

from caml.data import CausalDataset
from caml.samplers import create_splitter

from ..search_space import NuisanceModelSpec
from .base import BaseTunerBackend

if TYPE_CHECKING:
    from caml.scorers import CateScorer


class OptunaBackend(BaseTunerBackend):
    """Optuna backend for CATE model selection."""

    def __init__(self, direction="minimize", sampler=None, **kwargs):
        self.direction = direction
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            direction=self.direction, sampler=self.sampler, **kwargs
        )

    def optimize(self, objective, n_trials: int, n_jobs=1, timeout=None, **kwargs):
        """Run Optuna optimization."""
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
        outcome_model: BaseEstimator,
        treatment_model: BaseEstimator,
        regression_model: BaseEstimator,
    ):
        """Create Optuna objective function."""
        from caml.registry.registry import AVAILABLE_CATE_ESTIMATORS

        def objective(trial):
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
            return np.mean(scores)

        return objective
