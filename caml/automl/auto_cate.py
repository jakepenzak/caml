import inspect
import logging

import numpy as np

from caml._generics.decorators import experimental, narrate
from caml.data.dataset import CausalDataset
from caml.logging import LOGO, configure_logging
from caml.nuisance import NuisanceTuner, NuisanceTunerSpec

from .backends.base import BaseTunerBackend, TunerBackend
from .backends.optuna import OptunaBackend
from .search_space import ConstantSpec, NuisanceModelSpec, StandardMLSpec

logger = logging.getLogger(__name__)


@experimental
class AutoCATE:
    def __init__(
        self,
        *,
        nuisance_time_budget_s: int = 300,
        nuisance_tuner_spec: NuisanceTunerSpec | None = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        candidate_cate_estimators: list[str] | str = "auto",
        cate_scorer: str = "RLoss",
        optimization_backend: TunerBackend | None = None,
        cv: int = 3,
        test_set_fraction: float = 0.2,
        random_state: int | None = None,
        verbose: int | None = 1,
    ):
        """AutoCATE is a high-level interface for automated CATE model selection and tuning."""
        from caml.registry import AVAILABLE_CATE_ESTIMATORS, AVAILABLE_CATE_SCORERS

        if verbose is not None:
            configure_logging(verbose=verbose)

        self.nuisance_time_budget_s = nuisance_time_budget_s
        self.nuisance_tuner_spec = (
            nuisance_tuner_spec
            if nuisance_tuner_spec is not None
            else NuisanceTunerSpec()
        )
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.candidate_cate_estimators = candidate_cate_estimators
        self.cate_scorer = cate_scorer
        self.optimization_backend = (
            optimization_backend
            if optimization_backend is not None
            else OptunaBackend()
        )
        self.cv = cv
        self.test_set_fraction = test_set_fraction
        self.random_state = random_state
        self.outcome_model_ = None
        self.treatment_model_ = None
        self.regression_model_ = None
        self._fitted = False

        if not isinstance(self.nuisance_tuner_spec, NuisanceTunerSpec):
            raise ValueError(
                f"nuisance_tuner_spec must be a NuisanceTunerSpec instance, got {type(self.nuisance_tuner_spec)}"
            )
        if not isinstance(self.optimization_backend, BaseTunerBackend):
            raise ValueError(
                f"optimization_backend must be a BaseTunerBackend instance, got {type(self.optimization_backend)}"
            )
        if not isinstance(self.cv, int) or self.cv < 2:
            raise ValueError(f"cv must be an integer >= 2, got {self.cv}")
        if not isinstance(self.test_set_fraction, float) or not (
            0 < self.test_set_fraction < 1
        ):
            raise ValueError(
                f"test_set_fraction must be a float in (0, 1), got {self.test_set_fraction}"
            )
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError(
                f"random_state must be an integer or None, got {type(self.random_state)}"
            )

        if isinstance(self.candidate_cate_estimators, list):
            for ce in self.candidate_cate_estimators:
                if ce not in AVAILABLE_CATE_ESTIMATORS.keys():
                    raise ValueError(
                        f"Invalid candidate estimator: {ce}. Must be one of {AVAILABLE_CATE_ESTIMATORS.keys()}."
                    )
        elif self.candidate_cate_estimators == "auto":
            pass  # Will be determined based on dataset compatibility during fit
        else:
            raise ValueError(
                f"Invalid candidate_cate_estimators: {self.candidate_cate_estimators}. Must be a list of estimator names or 'auto'."
            )

        if self.cate_scorer not in AVAILABLE_CATE_SCORERS.keys():
            raise ValueError(
                f"Invalid cate_scorer: {self.cate_scorer}. Must be one of {AVAILABLE_CATE_SCORERS.keys()}."
            )

    @narrate(preamble=LOGO, epilogue=None)
    def fit(
        self,
        data: CausalDataset,
        use_cached_nuisance_models: bool = True,
    ):
        from caml.estimators.standard_ml import AVAILABLE_STANDARD_ML_ESTIMATORS
        from caml.registry import AVAILABLE_CATE_ESTIMATORS, EstimatorFamily

        ## Validate inputs
        self._validate(data)

        ## Create candidate estimator list if set to "auto"
        if self.candidate_cate_estimators == "auto":
            DEFAULT_FAMILIES = [
                EstimatorFamily.DR,
                EstimatorFamily.META,
                EstimatorFamily.DML,
            ]
            self.candidate_cate_estimators = [
                ce
                for ce in AVAILABLE_CATE_ESTIMATORS.keys()
                if AVAILABLE_CATE_ESTIMATORS[ce]["family"] in DEFAULT_FAMILIES
                and AVAILABLE_CATE_ESTIMATORS[ce]["estimator"].is_compatible_with(data)
            ]

        ## Hold out Test Set
        rng = np.random.default_rng(self.random_state)
        n = len(data.X)
        self.train_indices = rng.choice(
            n,
            size=int(n * (1 - self.test_set_fraction)),
            replace=False,
        )
        self.test_indices = np.setdiff1d(np.arange(n), self.train_indices)

        train_data = data.sample(self.train_indices)
        test_data = data.sample(self.test_indices)

        ## Fit Nuisance Models (if needed)
        need_regression_model, need_outcome_model, need_treatment_model = (
            self._determine_nuisance_requirements(
                self.candidate_cate_estimators, self.cate_scorer
            )
        )

        if use_cached_nuisance_models:
            if need_regression_model and self.regression_model_ is not None:
                need_regression_model = False
            if need_outcome_model and self.outcome_model_ is not None:
                need_outcome_model = False
            if need_treatment_model and self.treatment_model_ is not None:
                need_treatment_model = False

        if not all(
            not need
            for need in [
                need_regression_model,
                need_outcome_model,
                need_treatment_model,
            ]
        ):
            self.nuisance_tuner_spec.fit_regression_model = need_regression_model
            self.nuisance_tuner_spec.fit_outcome_model = need_outcome_model
            self.nuisance_tuner_spec.fit_treatment_model = need_treatment_model

            tuner = NuisanceTuner(
                time_budget=self.nuisance_time_budget_s, seed=self.random_state
            )
            tuner.fit(train_data, self.nuisance_tuner_spec)

            self.treatment_model_ = tuner.treatment_model_
            self.outcome_model_ = tuner.outcome_model_
            self.regression_model_ = tuner.regression_model_
        else:
            logger.info(
                "All required nuisance models are already fit and cached. Skipping nuisance tuning."
            )

        ## Run CATE Estimator Optimization
        scorer = self._get_scorer(self.cate_scorer)
        objective = self.optimization_backend.create_objective(
            data=train_data,
            scorer=scorer,
            candidate_cate_estimators=self.candidate_cate_estimators,
            cv=self.cv,
            outcome_model=self.outcome_model_,
            treatment_model=self.treatment_model_,
            regression_model=self.regression_model_,
        )

        self.study = self.optimization_backend.optimize(
            objective=objective, n_trials=self.n_trials, n_jobs=self.n_jobs
        )

        self.best_estimator_name_ = self.study.best_params["estimator"]
        self.best_estimator_ = AVAILABLE_CATE_ESTIMATORS[self.best_estimator_name_][
            "estimator"
        ]()

        for param in self.best_estimator_.default_search_space:
            if isinstance(param, NuisanceModelSpec):
                if param.model_type == "regression":
                    self.best_estimator_.set_params(
                        **{param.name: self.regression_model_}
                    )
                elif param.model_type == "outcome":
                    self.best_estimator_.set_params(**{param.name: self.outcome_model_})
                else:
                    self.best_estimator_.set_params(
                        **{param.name: self.treatment_model_}
                    )
            elif isinstance(param, StandardMLSpec):
                best_standard_ml_estimator_name = self.study.best_params[
                    f"{self.best_estimator_name_}__{param.name}"
                ]
                standard_ml_estimator = AVAILABLE_STANDARD_ML_ESTIMATORS[
                    best_standard_ml_estimator_name
                ]
                estimator = standard_ml_estimator._regressor_class()
                for inner_param in standard_ml_estimator.default_search_space:
                    if isinstance(inner_param, ConstantSpec):
                        val = inner_param.value
                    else:
                        val = self.study.best_params[
                            f"{self.best_estimator_name_}__{param.name}__{inner_param.name}"
                        ]
                    estimator.set_params(**{inner_param.name: val})
                self.best_estimator_.set_params(**{param.name: estimator})
            elif isinstance(param, ConstantSpec):
                self.best_estimator_.set_params(**{param.name: param.value})
            else:
                self.best_estimator_.set_params(
                    **{
                        param.name: self.study.best_params[
                            f"{self.best_estimator_name_}__{param.name}"
                        ]
                    }
                )

        ## Fit best estimator on full training data and evaluate on test set
        ## Break this out into a seperate method so users can evaluate on test set with different scorers after refitting best model on full dataset if desired
        self.best_estimator_.fit(train_data)
        scorer.normalized = True
        self.test_score_ = scorer(self.best_estimator_, test_data)
        logger.info(
            f"Best estimator: {self.best_estimator_name_} with normalized test score: {self.test_score_:.4f}"
        )
        self._fitted = True

    def refit_final(self, data: CausalDataset):
        """Refit the best estimator on the full dataset."""
        if not self._fitted:
            raise ValueError("Must call fit() before refitting final model.")
        self.best_estimator_.fit(data)
        logger.info("Best estimator refit on full dataset.")

    def _validate(self, data: CausalDataset):
        """Validate inputs to fit method."""
        from caml.registry import AVAILABLE_CATE_ESTIMATORS, AVAILABLE_CATE_SCORERS

        if not isinstance(data, CausalDataset):
            raise ValueError(f"data must be a CausalDataset instance, got {type(data)}")
        if isinstance(self.candidate_cate_estimators, list):
            for ce in self.candidate_cate_estimators:
                if not AVAILABLE_CATE_ESTIMATORS[ce]["estimator"].is_compatible_with(
                    data
                ):
                    raise ValueError(
                        f"Estimator {ce} is not compatible with the provided dataset. Check the estimator's compatibility requirements."
                    )
        elif self.candidate_cate_estimators == "auto":
            pass  # Will be determined based on dataset compatibility during fit
        else:
            raise ValueError(
                f"Invalid candidate_cate_estimators: {self.candidate_cate_estimators}. Must be a list of estimator names or 'auto'."
            )

        if not AVAILABLE_CATE_SCORERS[self.cate_scorer]["scorer"].is_compatible_with(
            data
        ):
            raise ValueError(
                f"CATE scorer {self.cate_scorer} is not compatible with the provided dataset. Check the scorer's compatibility requirements."
            )

        if (
            self.optimization_backend.direction == "minimize"
            and AVAILABLE_CATE_SCORERS[self.cate_scorer][
                "scorer"
            ].capabilities.greater_is_better
        ):
            raise ValueError(
                f"CATE scorer {self.cate_scorer} is a 'greater is better' metric, but the optimization backend is set to minimize. Either change the optimization direction or choose a different scorer."
            )

    @staticmethod
    def _determine_nuisance_requirements(candidate_cate_estimators, cate_scorer):
        """Determine which nuisance models need to be fit based on the requirements of the candidate estimators and CATE scorer."""
        from caml.registry import AVAILABLE_CATE_ESTIMATORS, AVAILABLE_CATE_SCORERS

        _need_regression_model = False
        _need_outcome_model = False
        _need_treatment_model = False
        for x in candidate_cate_estimators:
            if AVAILABLE_CATE_ESTIMATORS[x][
                "estimator"
            ].capabilities.requires_regression_model:
                _need_regression_model = True
            if AVAILABLE_CATE_ESTIMATORS[x][
                "estimator"
            ].capabilities.requires_treatment_model:
                _need_treatment_model = True
            if AVAILABLE_CATE_ESTIMATORS[x][
                "estimator"
            ].capabilities.requires_outcome_model:
                _need_outcome_model = True

        if AVAILABLE_CATE_SCORERS[cate_scorer][
            "scorer"
        ].capabilities.requires_regression_model:
            _need_regression_model = True
        if AVAILABLE_CATE_SCORERS[cate_scorer][
            "scorer"
        ].capabilities.requires_treatment_model:
            _need_treatment_model = True
        if AVAILABLE_CATE_SCORERS[cate_scorer][
            "scorer"
        ].capabilities.requires_outcome_model:
            _need_outcome_model = True

        return _need_regression_model, _need_outcome_model, _need_treatment_model

    def _get_scorer(self, scorer):
        from caml.registry import AVAILABLE_CATE_SCORERS

        scorer = AVAILABLE_CATE_SCORERS[scorer]["scorer"]

        kwargs = {}
        parameters = list(inspect.signature(scorer.__init__).parameters)
        if "cv" in parameters:
            kwargs["cv"] = self.cv
        if "random_state" in parameters:
            kwargs["random_state"] = self.random_state

        if scorer.capabilities.requires_regression_model:
            if "regression_model" not in parameters:
                raise ValueError(
                    f"Scorer {scorer} requires a regression model but does not accept one as an argument. Check the scorer's implementation."
                )
            kwargs["regression_model"] = self.regression_model_
        if scorer.capabilities.requires_outcome_model:
            if "outcome_model" not in parameters:
                raise ValueError(
                    f"Scorer {scorer} requires an outcome model but does not accept one as an argument. Check the scorer's implementation."
                )
            kwargs["outcome_model"] = self.outcome_model_
        if scorer.capabilities.requires_treatment_model:
            if "treatment_model" not in parameters:
                raise ValueError(
                    f"Scorer {scorer} requires a treatment model but does not accept one as an argument. Check the scorer's implementation."
                )
            kwargs["treatment_model"] = self.treatment_model_

        return scorer(**kwargs)
