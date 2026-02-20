"""Optuna backend implementation."""

import optuna

from ..search_space import NuisanceModelSpec
from .base import BaseTunerBackend


class OptunaBackend(BaseTunerBackend):
    """Optuna backend for CATE model selection."""

    def __init__(self, direction="minimize", sampler=None, **kwargs):
        self.direction = direction
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.kwargs = kwargs

    def optimize(self, objective, n_trials: int, n_jobs=1):
        """Run Optuna optimization."""
        study = optuna.create_study(
            direction=self.direction, sampler=self.sampler, **self.kwargs
        )

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        return study

    def create_objective(
        self,
        scorer,
        candidate_estimators,
        data,
        outcome_model,
        treatment_model,
        regression_model,
    ):
        """Create Optuna objective function."""

        def objective(trial):
            estimator_name = trial.suggest_categorical(
                "estimator", candidate_estimators.keys()
            )
            estimator = candidate_estimators[estimator_name]["estimator"]()
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
            estimator.fit(data)
            return scorer(estimator, data)

        return objective
