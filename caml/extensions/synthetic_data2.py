import math
from typing import Callable

import numpy as np
import pandas as pd
from doubleml.datasets import (
    make_heterogeneous_data,
    make_plr_CCDDHNR2018,
    make_plr_turrell2018,
)
from numpy.random import choice
from scipy.linalg import toeplitz
from typeguard import typechecked

from ..generics import cls_typechecked


# TODO: This class needs some serious cleaning up
@cls_typechecked
class CamlSyntheticDataGenerator:
    def __init__(
        self,
        n_obs: int = 10_000,
        n_cont_outcomes: int = 1,
        n_binary_outcomes: int = 0,
        n_cont_treatments: int = 0,
        n_binary_treatments: int = 1,
        n_discrete_treatments: int = 0,
        n_cont_confounders: int = 2,
        n_binary_confounders: int = 2,
        n_discrete_confounders: int = 0,
        n_cont_heterogeneity_covariates: int = 2,
        n_binary_heterogeneity_covariates: int = 2,
        n_discrete_heterogeneity_covariates: int = 0,
        n_heterogeneity_confounders: int = 0,
        stddev_outcome_noise: float = 1.0,
        stddev_treatment_noise: float = 1.0,
        causal_model_functional_form: str = "partially_linear",
        n_nonlinear_transformations: int | None = None,
        n_nonlinear_interactions: int | None = None,
        seed: int | None = None,
    ):
        valid_functional_forms = [
            "partially_linear",
            "fully_nonlinear",
            "fully_linear",
        ]

        assert (
            causal_model_functional_form in valid_functional_forms
        ), f"Invalid functional form. Must be choice of {valid_functional_forms}."

        self._n_obs = n_obs
        self._n_cont_outcomes = n_cont_outcomes
        self._n_binary_outcomes = n_binary_outcomes
        self._n_cont_treatments = n_cont_treatments
        self._n_binary_treatments = n_binary_treatments
        self._n_discrete_treatments = n_discrete_treatments
        self._n_cont_confounders = n_cont_confounders
        self._n_binary_confounders = n_binary_confounders
        self._n_discrete_confounders = n_discrete_confounders
        self._n_cont_heterogeneity_covariates = n_cont_heterogeneity_covariates
        self._n_binary_heterogeneity_covariates = n_binary_heterogeneity_covariates
        self._n_discrete_heterogeneity_covariates = n_discrete_heterogeneity_covariates
        self._n_heterogeneity_confounders = n_heterogeneity_confounders
        self._stddev_outcome_noise = stddev_outcome_noise
        self._stddev_treatment_noise = stddev_treatment_noise
        self._causal_model_functional_form = causal_model_functional_form
        self._n_nonlinear_transformations = n_nonlinear_transformations
        self._n_nonlinear_interactions = n_nonlinear_interactions
        self._seed = seed if seed is not None else np.random.randint(1, 1000)
        self._rng = np.random.default_rng(seed)

        self._generate_data()

    def _generate_data(self):
        # Generate confounders
        confounders = self._generate_confounder_variables()

        # Generate heterogeneity variables
        heterogeneity_variables = self._generate_heterogenous_variables()

        # Generate treatment variables
        treatments, treatments_dgp = self._generate_treatment_variables(
            confounders=confounders,
            heterogeneity_variables=heterogeneity_variables,
        )

        # Generate outcome variables
        outcomes, cates, outcomes_dgp = self._generate_outcome_variables(
            confounders=confounders,
            heterogeneity_variables=heterogeneity_variables,
            treatments=treatments,
        )

        # Combine variables into single dataframe
        synthetic_data = pd.concat(
            [confounders, heterogeneity_variables, treatments, outcomes], axis=1
        )

        # Prettify CATEs and ATEs report
        cate_df, ate_df = self._treatment_effect_report(cates)

        self.df = synthetic_data
        self.cates = cate_df
        self.ates = ate_df
        self.dgp = treatments_dgp | outcomes_dgp

    def _generate_confounder_variables(self) -> pd.DataFrame:
        """Generate confounder variables."""

        confounders = {}

        for i in range(self._n_cont_confounders):
            confounders[f"W{i+1}_continuous"] = self._generate_random_variable(
                n_obs=self._n_obs,
                var_type="continuous",
                rng=self._rng,
            )
        for i in range(self._n_binary_confounders):
            confounders[f"W{i+1}_binary"] = self._generate_random_variable(
                n_obs=self._n_obs,
                var_type="binary",
                rng=self._rng,
            )
        for i in range(self._n_discrete_confounders):
            confounders[f"W{i+1}_discrete"] = self._generate_random_variable(
                n_obs=self._n_obs,
                var_type="discrete",
                rng=self._rng,
            )

        return pd.DataFrame(confounders)

    def _generate_heterogenous_variables(self) -> pd.DataFrame:
        """Generate treatment heterogeneity inducing covariates variables."""

        heterogeneity_variables = {}

        for i in range(self._n_cont_heterogeneity_covariates):
            heterogeneity_variables[f"X{i+1}_continuous"] = (
                self._generate_random_variable(
                    n_obs=self._n_obs,
                    var_type="continuous",
                    rng=self._rng,
                )
            )
        for i in range(self._n_binary_heterogeneity_covariates):
            heterogeneity_variables[f"X{i+1}_binary"] = self._generate_random_variable(
                n_obs=self._n_obs,
                var_type="binary",
                rng=self._rng,
            )
        for i in range(self._n_discrete_heterogeneity_covariates):
            heterogeneity_variables[f"X{i+1}_discrete"] = (
                self._generate_random_variable(
                    n_obs=self._n_obs,
                    var_type="discrete",
                    rng=self._rng,
                )
            )

        return pd.DataFrame(heterogeneity_variables)

    def _generate_treatment_variables(
        self, confounders: pd.DataFrame, heterogeneity_variables: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Generate the treatment variables"""

        treatments = {}
        dgp = {}
        subset_heterogeneity = heterogeneity_variables.sample(
            n=self._n_heterogeneity_confounders, axis=1, random_state=self._seed
        )

        all_confounders = pd.concat([confounders, subset_heterogeneity], axis=1)

        if self._causal_model_functional_form == "fully_linear":
            dgp_type = "linear"
            for i in range(self._n_cont_treatments):
                col_name = f"T{i+1}_continuous"
                treatments[col_name], dgp[col_name] = self._dgp(
                    covariates=all_confounders,
                    stddev_err=self._stddev_treatment_noise,
                    n_obs=self._n_obs,
                    dep_type="continuous",
                    dgp_type=dgp_type,
                )
            for i in range(self._n_binary_treatments):
                col_name = f"T{i+1}_binary"
                treatments[col_name], dgp[col_name] = self._dgp(
                    covariates=all_confounders,
                    stddev_err=self._stddev_treatment_noise,
                    n_obs=self._n_obs,
                    dep_type="binary",
                    dgp_type=dgp_type,
                )
            for i in range(self._n_discrete_treatments):
                col_name = f"T{i+1}_discrete"
                treatments[col_name], dgp[col_name] = self._dgp(
                    covariates=all_confounders,
                    stddev_err=self._stddev_treatment_noise,
                    n_obs=self._n_obs,
                    dep_type="discrete",
                    dgp_type=dgp_type,
                )
        elif self._causal_model_functional_form in [
            "partially_linear",
            "fully_nonlinear",
        ]:
            dgp_type = "nonlinear"
            for i in range(self._n_cont_treatments):
                col_name = f"T{i+1}_continuous"
                treatments[col_name], dgp[col_name] = self._dgp(
                    covariates=all_confounders,
                    stddev_err=self._stddev_treatment_noise,
                    n_obs=self._n_obs,
                    dep_type="continuous",
                    dgp_type=dgp_type,
                    n_nonlinear_transformations=self._n_nonlinear_transformations,
                    n_nonlinear_interactions=self._n_nonlinear_interactions,
                )
            for i in range(self._n_binary_treatments):
                col_name = f"T{i+1}_binary"
                treatments[col_name], dgp[col_name] = self._dgp(
                    covariates=all_confounders,
                    stddev_err=self._stddev_treatment_noise,
                    n_obs=self._n_obs,
                    dep_type="binary",
                    dgp_type=dgp_type,
                    n_nonlinear_transformations=self._n_nonlinear_transformations,
                    n_nonlinear_interactions=self._n_nonlinear_interactions,
                )
            for i in range(self._n_discrete_treatments):
                col_name = f"T{i+1}_discrete"
                treatments[col_name], dgp[col_name] = self._dgp(
                    covariates=all_confounders,
                    stddev_err=self._stddev_treatment_noise,
                    n_obs=self._n_obs,
                    dep_type="discrete",
                    dgp_type=dgp_type,
                    n_nonlinear_transformations=self._n_nonlinear_transformations,
                    n_nonlinear_interactions=self._n_nonlinear_interactions,
                )
        else:
            raise ValueError(
                f"Invalid causal model functional form: {self._causal_model_functional_form}"
            )

        return pd.DataFrame(treatments), dgp

    def _generate_outcome_variables(
        self,
        confounders: pd.DataFrame,
        heterogeneity_variables: pd.DataFrame,
        treatments: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict, dict]:
        """Generate the outcome variables"""

        outcomes = {}
        cates = {}
        dgp = {}

        if self._causal_model_functional_form in ["fully_linear", "partially_linear"]:
            # Generate Interaction terms
            interactions_np = (
                treatments.to_numpy()[:, :, None]
                * heterogeneity_variables.to_numpy()[:, None, :]
            ).reshape(self._n_obs, -1)
            interactions = pd.DataFrame(
                interactions_np,
                columns=[
                    f"int_{t}_{x}"
                    for t in treatments.columns
                    for x in heterogeneity_variables.columns
                ],
            )
            all_features = pd.concat(
                [confounders, heterogeneity_variables, treatments, interactions], axis=1
            )
            if self._causal_model_functional_form == "fully_linear":
                dgp_type = "linear"
                for i in range(self._n_cont_outcomes):
                    col_name = f"Y{i+1}_continuous"
                    outcomes[col_name], cates[col_name], dgp[col_name] = self._dgp(
                        covariates=all_features,
                        stddev_err=self._stddev_outcome_noise,
                        n_obs=self._n_obs,
                        dep_type="continuous",
                        dgp_type=dgp_type,
                        return_treatment_effects=True,
                    )
                for i in range(self._n_binary_outcomes):
                    col_name = f"Y{i+1}_binary"
                    outcomes[col_name], cates[col_name], dgp[col_name] = self._dgp(
                        covariates=all_features,
                        stddev_err=self._stddev_outcome_noise,
                        n_obs=self._n_obs,
                        dep_type="binary",
                        dgp_type=dgp_type,
                        return_treatment_effects=True,
                    )
            else:
                pass
        elif self._causal_model_functional_form in ["fully_nonlinear"]:
            dgp_type = "nonlinear"
            all_features = pd.concat(
                [confounders, heterogeneity_variables, treatments], axis=1
            )
            for i in range(self._n_cont_outcomes):
                col_name = f"Y{i+1}_continuous"
                outcomes[col_name], cates[col_name], dgp[col_name] = self._dgp(
                    covariates=all_features,
                    stddev_err=self._stddev_outcome_noise,
                    n_obs=self._n_obs,
                    dep_type="continuous",
                    dgp_type=dgp_type,
                    return_treatment_effects=True,
                    n_nonlinear_transformations=self._n_nonlinear_transformations,
                    n_nonlinear_interactions=self._n_nonlinear_interactions,
                )
            for i in range(self._n_binary_outcomes):
                col_name = f"Y{i+1}_binary"
                outcomes[col_name], cates[col_name], dgp[col_name] = self._dgp(
                    covariates=all_features,
                    stddev_err=self._stddev_outcome_noise,
                    n_obs=self._n_obs,
                    dep_type="binary",
                    dgp_type=dgp_type,
                    return_treatment_effects=True,
                    n_nonlinear_transformations=self._n_nonlinear_transformations,
                    n_nonlinear_interactions=self._n_nonlinear_interactions,
                )
        else:
            raise ValueError(
                f"Invalid causal model functional form: {self._causal_model_functional_form}"
            )

        return pd.DataFrame(outcomes), cates, dgp

    def _dgp(
        self,
        covariates: pd.DataFrame,
        stddev_err: float,
        n_obs: int,
        dep_type: str,
        dgp_type: str,
        return_treatment_effects: bool = False,
        n_nonlinear_transformations: int | None = None,
        n_nonlinear_interactions: int | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame] | tuple[np.ndarray, dict, pd.DataFrame]:
        """DGP for treatments and outcomes"""

        valid_dep_types = ["continuous", "binary", "discrete"]
        valid_dgp_types = ["linear", "partially_linear", "nonlinear"]
        assert (
            dep_type in valid_dep_types
        ), f"Invalid dep type: {dep_type}. Choose from {valid_dep_types}."
        assert (
            dgp_type in valid_dgp_types
        ), f"Invalid dgp type: {dgp_type}. Choose from {valid_dgp_types}."

        n_features = covariates.shape[1]

        if dep_type == "continuous":
            if dgp_type == "linear":
                params = self._rng.uniform(low=-3, high=3, size=n_features)

                noise = self._rng.normal(0, stddev_err, size=n_obs)

                def f(x):
                    return x @ params + noise

                linear_score = f(covariates.values)

            elif dgp_type == "nonlinear":
                transformed_covariates = self._apply_random_nonlinearities(
                    data=covariates,
                    n_transforms=n_nonlinear_transformations
                    if n_nonlinear_transformations
                    else n_features,
                    n_interactions=n_nonlinear_interactions
                    if n_nonlinear_interactions
                    else 5,
                    seed=self._seed,
                )
                params = self._rng.uniform(
                    low=-3, high=3, size=transformed_covariates.shape[1]
                )

                noise = self._rng.normal(0, stddev_err, size=n_obs)

                def f(x):
                    return x @ params + noise

                linear_score = f(transformed_covariates.values)

            else:
                raise ValueError("Not ready")

            dep = linear_score

        elif dep_type == "binary":
            if dgp_type == "linear":
                params = self._rng.uniform(low=-3, high=3, size=n_features)
                noise = self._rng.normal(0, stddev_err, size=n_obs)

                def f(x):
                    linear_scores = x @ params + noise
                    return self._sigmoid(linear_scores)

                probabilities = f(covariates.values)

            elif dgp_type == "nonlinear":
                transformed_covariates = self._apply_random_nonlinearities(
                    data=covariates,
                    n_transforms=n_nonlinear_transformations
                    if n_nonlinear_transformations
                    else n_features,
                    n_interactions=n_nonlinear_interactions
                    if n_nonlinear_interactions
                    else 5,
                    seed=self._seed,
                )
                params = self._rng.uniform(
                    low=-3, high=3, size=transformed_covariates.shape[1]
                )

                noise = self._rng.normal(0, stddev_err, size=n_obs)

                def f(x):
                    nonlinear_scores = x @ params + noise
                    return self._sigmoid(nonlinear_scores)

                probabilities = f(transformed_covariates.values)

            else:
                raise ValueError("Not ready")

            dep = self._rng.binomial(1, probabilities)

        elif dep_type == "discrete":
            if dgp_type == "linear":
                n_categories = self._rng.choice(range(3, 6))
                params = self._rng.uniform(
                    low=-3, high=3, size=(n_features, n_categories)
                )
                noise = self._rng.normal(0, stddev_err, size=(n_obs, n_categories))

                def f(x):
                    linear_scores = x @ params + noise
                    return self._softmax(linear_scores)

                probabilities = f(covariates.values)

            elif dgp_type == "nonlinear":
                transformed_covariates = self._apply_random_nonlinearities(
                    data=covariates,
                    n_transforms=n_nonlinear_transformations
                    if n_nonlinear_transformations
                    else n_features,
                    n_interactions=n_nonlinear_interactions
                    if n_nonlinear_interactions
                    else 5,
                    seed=self._seed,
                )
                n_categories = self._rng.choice(range(3, 6))
                params = self._rng.uniform(
                    low=-3, high=3, size=(transformed_covariates.shape[1], n_categories)
                )
                noise = self._rng.normal(0, stddev_err, size=(n_obs, n_categories))

                def f(x):
                    nonlinear_scores = x @ params + noise
                    return self._softmax(nonlinear_scores)

                probabilities = f(transformed_covariates.values)

            else:
                raise ValueError("Not ready")

            dep = np.array(
                [
                    self._rng.choice(range(n_categories), p=prob)
                    for prob in probabilities
                ]
            )

        else:
            raise ValueError("Invalid dependent variable type.")

        dgp = {}
        dgp["covariates"] = (
            list(covariates.columns)
            if dgp_type == "linear"
            else list(transformed_covariates.columns)
        )
        dgp["params"] = params
        if dgp["params"].ndim > 1:
            for i in range(dgp["params"].ndim):
                dgp[f"cat_{i+1}_params"] = dgp["params"][:, i]
            del dgp["params"]

        dgp["transformation"] = (
            "Sigmoid"
            if dep_type == "binary"
            else "Softmax"
            if dep_type == "discrete"
            else "None"
        )

        if return_treatment_effects:
            cates = self._compute_treatment_effects(
                f,
                covariates,
                dgp_type,
            )
            return dep, cates, pd.DataFrame(dgp)
        return dep, pd.DataFrame(dgp)

    def _compute_treatment_effects(
        self,
        f: Callable,
        data: pd.DataFrame,
        dgp_type: str,
    ) -> dict:
        """Compute treatment effects."""

        cates = {}
        for t in [c for c in data.columns if c.count("_") == 1 and "T" in c]:
            if "continuous" in t:
                levels = ["cont"]
            elif "binary" in t:
                levels = [1]
            elif "discrete" in t:
                levels = data[t].unique().tolist()
            else:
                raise ValueError("Invalid treatment type.")

            cates[t] = self._compute_potential_outcome_differences(
                f,
                data,
                t,
                dgp_type,
                levels=levels,
            )

        return cates

    def _compute_potential_outcome_differences(
        self,
        f: Callable,
        data: pd.DataFrame,
        wrt: str,
        dgp_type: str,
        levels: list,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Compute potential outcome differences."""

        cates = {}
        for lev in levels:
            if lev == 0:
                pass
            else:
                data_treat = data.copy()
                data_control = data.copy()

                if lev == "cont":
                    data_treat[wrt] = data_treat[wrt] + 1
                else:
                    data_treat[wrt] = lev
                    data_control[wrt] = 0

                if dgp_type == "nonlinear":
                    data_treat_transformed = self._apply_random_nonlinearities(
                        data_treat,
                        n_transforms=self._n_nonlinear_transformations
                        if self._n_nonlinear_transformations
                        else data_treat.shape[1],
                        n_interactions=self._n_nonlinear_interactions
                        if self._n_nonlinear_interactions
                        else 5,
                        seed=self._seed,
                    )
                    data_control_transformed = self._apply_random_nonlinearities(
                        data_control,
                        n_transforms=self._n_nonlinear_transformations
                        if self._n_nonlinear_transformations
                        else data_control.shape[1],
                        n_interactions=self._n_nonlinear_interactions
                        if self._n_nonlinear_interactions
                        else 5,
                        seed=self._seed,
                    )

                    cates[f"{lev}_v_0"] = f(data_treat_transformed.values) - f(
                        data_control_transformed.values
                    )
                else:
                    for interaction in [
                        c for c in data.columns if wrt in c and "int" in c
                    ]:
                        covariate = interaction.split(f"{wrt}_")[1]
                        data_treat[interaction] = (
                            data_treat[covariate] * data_treat[wrt]
                        )
                        data_control[interaction] = (
                            data_control[covariate] * data_control[wrt]
                        )

                    cates[f"{lev}_v_0"] = f(data_treat.values) - f(data_control.values)

        if len(cates) == 1:
            return list(cates.values())[0]
        else:
            return cates

    @staticmethod
    def _generate_random_variable(
        n_obs: int,
        var_type: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates a random variable of n observations from randomly selected
        distribution given a data type.

        Parameters
        ----------
        n_obs
            Number of observations.
        var_type
            Type of the variable to generate, choose from "continuous", "binary" or "discrete".
        rng
            Numpy random number generator.

        Returns
        -------
        np.ndarray
            Generated random variable.

        """

        valid_types = ["continuous", "binary", "discrete"]
        assert (
            var_type in valid_types
        ), f"Invalid type: {var_type}. Choose from {valid_types}."

        if var_type == "continuous":
            distributions = [
                "normal",
                "uniform",
                "exponential",
                "gamma",
                "beta",
                "laplace",
            ]

            dist = rng.choice(distributions)

            if dist == "normal":
                mean, std = rng.uniform(-5, 5), rng.uniform(0.5, 2)
                res = rng.normal(mean, std, n_obs)
            elif dist == "uniform":
                low, high = rng.uniform(-10, 0), rng.uniform(0, 10)
                res = rng.uniform(low, high, n_obs)
            elif dist == "exponential":
                scale = rng.uniform(1, 3)
                res = rng.exponential(scale, n_obs)
            elif dist == "gamma":
                shape, scale = rng.uniform(1, 3), rng.uniform(1, 3)
                res = rng.gamma(shape, scale, n_obs)
            elif dist == "beta":
                a, b = rng.uniform(1, 3), rng.uniform(1, 3)
                res = rng.beta(a, b, n_obs)
            elif dist == "laplace":
                loc, scale = rng.uniform(-5, 5), rng.uniform(0.5, 2)
                res = rng.laplace(loc, scale, n_obs)
            else:
                raise ValueError("Invalid distribution")

        elif var_type == "binary":
            p = rng.uniform(0.1, 0.9)
            res = rng.binomial(1, p, n_obs)

        elif var_type == "discrete":
            distributions = ["poisson", "geometric", "multinomial", "uniform"]

            dist = rng.choice(distributions)

            if dist == "poisson":
                lam = rng.uniform(1, 10)
                res = rng.poisson(lam, n_obs)

            elif dist == "geometric":
                p = rng.uniform(0.1, 0.9)
                res = rng.geometric(p, n_obs)

            elif dist == "multinomial":
                n_categories = rng.choice(range(2, 7))
                probs = rng.dirichlet(np.ones(n_categories))
                res = rng.choice(range(n_categories), size=n_obs, p=probs)

            elif dist == "uniform":
                n_categories = rng.choice(range(2, 7))
                res = rng.choice(range(0, n_categories), size=n_obs)
            else:
                raise ValueError("Invalid distribution")

        else:
            raise ValueError("Invalid variable type.")

        return res

    @staticmethod
    def _treatment_effect_report(
        cates: dict[str, dict],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a report on treatment effects."""

        dict_effects = {}
        for outcome, effects in cates.items():
            for treatment, values in effects.items():
                var = f"CATE_of_{treatment}_on_{outcome}"
                if isinstance(values, dict):
                    for levels, results in values.items():
                        var_lev = var + f"_level_{levels}"
                        dict_effects[var_lev] = results
                else:
                    dict_effects[var] = values

        cate_df = pd.DataFrame(dict_effects)

        ate_df = cate_df.mean(axis=0).reset_index()
        ate_df.columns = ["Treatment", "ATE"]
        ate_df["Treatment"] = ate_df["Treatment"].str.replace("CATE_of_", "")

        return cate_df, ate_df

    @staticmethod
    def _apply_random_nonlinearities(
        data: pd.DataFrame,
        n_transforms: int = 10,
        n_interactions: int = 5,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Apply a random set of nonlinear transformations to the input features.
        """
        np.random.seed(seed)

        if data.shape[1] > 0:
            transformed_data = data.copy()
            for _ in range(n_transforms):
                transform = np.random.choice(
                    [
                        "sin",
                        "cos",
                        "square",
                        " log",
                        "2D-interaction",
                        "sqrt",
                    ]
                )
                col_name = np.random.choice(data.columns, size=2, replace=True)

                if transform == "sin":
                    transformed_data[f"sin_{col_name[0]}"] = np.sin(
                        transformed_data[col_name[0]]
                    )
                elif transform == "cos":
                    transformed_data[f"cos_{col_name[0]}"] = np.cos(
                        transformed_data[col_name[0]]
                    )
                elif transform == "square":
                    transformed_data[f"square_{col_name[0]}"] = (
                        transformed_data[col_name[0]] ** 2
                    )
                elif transform == "log":
                    transformed_data[f"log_{col_name[0]}"] = np.log(
                        np.abs(transformed_data[col_name[0]]) + 1
                    )
                elif transform == "2D-interaction":
                    for c in col_name:
                        if c.count("_") == 1 and "T" in c:
                            pass
                        else:
                            transformed_data[f"2Dint_{col_name[0]}_{col_name[1]}"] = (
                                transformed_data[col_name[0]]
                                * transformed_data[col_name[1]]
                            )
                elif transform == "sqrt":
                    transformed_data[f"sqrt_{col_name[0]}"] = np.sqrt(
                        np.abs(transformed_data[col_name[0]])
                    )

            # Add explicit interaction terms for heterogeneity
            transformed_data_w_heterogeneity = transformed_data.copy()
            for t in [
                c for c in transformed_data.columns if "T" in c and c.count("_") == 1
            ]:
                for _ in range(n_interactions):
                    available_cols = [
                        c
                        for c in transformed_data.columns
                        if not ("T" in c and c.count("_") == 1)
                        and ("X" in c and "W" not in c)
                    ]
                    col = np.random.choice(available_cols, size=1)[0]

                    transformed_data_w_heterogeneity[f"int_{t}_{col}"] = (
                        transformed_data[t] * transformed_data[col]
                    )

            return transformed_data_w_heterogeneity
        else:
            return data

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid"""

        result = np.zeros_like(x, dtype=float)

        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

        return result

    @staticmethod
    def _softmax(x):
        """Numerically stable softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


@typechecked
def make_partially_linear_dataset_simple(
    n_obs: int = 1000,
    n_confounders: int = 5,
    dim_heterogeneity: int = 2,
    binary_treatment: bool = True,
    seed: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Simulate data generating process from a partially linear model with a simple 1 or 2 dimensional CATE function. The outcome is continuous and the treatment can be binary or continuous.
    The dataset is generated using the `make_heterogeneous_data` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is, in the case of dim_heterogeneity=1:

    $$
    y_i= \\tau (x_0) d_i + g(\mathbf{X_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{X_i})+\eta_i
    $$

    or, in the case of dim_heterogeneity=2:

    $$
    y_i= \\tau (x_0,x_1) d_i + g(\mathbf{X_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{X_i})+\eta_i
    $$

    where $y_i$ is the outcome, $d_i$ is the treatment, $\mathbf{X_i}$ are the confounders, $\epsilon_i$ and $\eta_i$ are the error terms, $\\tau$ is the CATE function, $g$ is the outcome function, and $f$ is the treatment function.

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    Here the ATE is defined as the average of the CATE function over all observations: $\mathbb{E}[\\tau (\cdot)]$

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        Xn((X))-->d((d));
        Xn((X))-->y((y));
        d((d))-->|"τ(x0,x1)"|y((y));

        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    n_confounders : int
        The number of confounders $\mathbf{X_i}$.
    dim_heterogeneity : int
        The dimension of the heterogeneity $x_0$ or $(x_0,x_1)$. Can only be 1 or 2.
    binary_treatment : bool
        Whether the treatment $d_i$ is binary or continuous.
    seed : int | None
        The seed to use for the random number generator.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and X are the confounders with a 1d or 2d subset utilized for heterogeneity.
    true_cates : numpy.ndarray
        The true conditional average treatment effects.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import make_partially_linear_dataset_simple
    df, true_cates, true_ate = make_partially_linear_dataset_simple(n_obs=1000,
                                                                    n_confounders=5,
                                                                    dim_heterogeneity=2,
                                                                    binary_treatment=True,
                                                                    seed=1)

    print(f"True CATES: {true_cates[:5]}")
    print(f"True ATE: {true_ate}")
    print(df.head())
    ```
    """

    if dim_heterogeneity not in [1, 2]:
        raise ValueError("dim_heterogeneity must be 1 or 2.")

    np.random.seed(seed)

    data = make_heterogeneous_data(
        n_obs=n_obs,
        p=n_confounders,
        support_size=n_confounders,
        n_x=dim_heterogeneity,
        binary_treatment=binary_treatment,
    )

    df = pd.DataFrame(data["data"])
    df.columns = [c.replace("X_", "X") for c in df.columns]
    true_cates = data["effects"]
    true_ate = true_cates.mean()
    return df, true_cates, true_ate


@typechecked
def make_partially_linear_dataset_constant(
    n_obs: int = 1000,
    ate: float = 4.0,
    n_confounders: int = 10,
    dgp: str = "make_plr_CCDDHNR2018",
    seed: int | None = None,
    **doubleml_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Simulate a data generating process from a partially linear model with a constant treatment effect (ATE only). The outcome and treatment are both continuous.
    The dataset is generated using the `make_plr_CCDDHNR2018` or `make_plr_turrell2018` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is:

    $$
    y_i= \\tau_0 d_i + g(\mathbf{W_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{W_i})+\eta_i
    $$

    where $y_i$ is the outcome, $d_i$ is the treatment, $\mathbf{W_i}$ are the confounders, $\epsilon_i$ and $\eta_i$ are the error terms, $\\tau_0$ is the ATE parameter, $g$ is the outcome function, and $f$ is the treatment function.

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        W((W))-->d((d));
        W((W))-->y((y));
        d((d))-->|"τ0"|y((y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    ate : float
        The average treatment effect $\\tau_0$.
    n_confounders : int
        The number of confounders $\mathbf{W_i}$ to generate.
    dgp : str
        The data generating process to use. Can be "make_plr_CCDDHNR20" or "make_plr_turrell2018".
    seed : int | None
        The seed to use for the random number generator.
    **doubleml_kwargs
        Additional keyword arguments to pass to the data generating process.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and W are the confounders.
    true_cates : numpy.ndarray
        The true conditional average treatment effects, which are all equal to the ATE here.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import make_partially_linear_dataset_constant
    df, true_cates, true_ate = make_partially_linear_dataset_constant(n_obs=1000,
                                                        ate=4.0,
                                                        n_confounders=10,
                                                        dgp="make_plr_CCDDHNR2018",
                                                        seed=1)

    print(f"True CATES: {true_cates[:5]}")
    print(f"True ATE: {true_ate}")
    print(df.head())
    ```
    """

    np.random.seed(seed)

    if dgp == "make_plr_CCDDHNR2018":
        df = make_plr_CCDDHNR2018(
            n_obs=n_obs,
            dim_x=n_confounders,
            alpha=ate,
            return_type="DataFrame",
            **doubleml_kwargs,
        )
    elif dgp == "make_plr_turrell2018":
        df = make_plr_turrell2018(
            n_obs=n_obs,
            dim_x=n_confounders,
            theta=ate,
            return_type="DataFrame",
            **doubleml_kwargs,
        )
    else:
        raise ValueError(
            "dgp must be 'make_plr_CCDDHNR2018' or 'make_plr_turrell2018'."
        )

    df.columns = [c.replace("X", "W") for c in df.columns if "X" in c] + ["y", "d"]

    true_ate = ate
    true_cates = np.full(n_obs, true_ate)

    return df, true_cates, true_ate


@typechecked
def make_fully_heterogeneous_dataset(
    n_obs: int = 1000,
    n_confounders: int = 5,
    theta: float = 4.0,
    seed: int | None = None,
    **doubleml_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Simulate data generating process from an interactive regression model with fully heterogenous treatment effects. The outcome is continuous and the treatment is binary.
    The dataset is generated using a modified version of `make_irm_data` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is:

    $$
    y_i= g(d_i,\mathbf{X_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{X_i})+\eta_i
    $$

    where $y_i$ is the outcome, $d_i$ is the treatment, $\mathbf{X_i}$ are the confounders utilized for full effect heterogeneity, $\epsilon_i$ and $\eta_i$ are the error terms, $g$ is the outcome function, and $f$ is the treatment function.

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    Note that the treatment effect is fully heterogenous, thus the CATE is defined as: $\\tau = \\mathbb{E}[g(1,\mathbf{X}) - g(0,\mathbf{X})|\mathbf{X}]$ for any $\mathbf{X}$.

    The ATE is defined as the average of the CATE function over all observations: $\mathbb{E}[\\tau (\cdot)]$

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        X((X))-->d((d));
        X((X))-->y((y));
        d((d))-->|"τ(X)"|y((y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    n_confounders : int
        The number of confounders $\mathbf{X_i}$ to generate (these are utilized fully for heterogeneity).
    theta : float
        The base parameter for the treatment effect. Note this differs from the ATE.
    seed : int | None
        The seed to use for the random number generator.
    **doubleml_kwargs
        Additional keyword arguments to pass to the data generating process.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and X are the confounders which are fully utilized for heterogeneity.
    true_cates : numpy.ndarray
        The true conditional average treatment effects.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import make_fully_heterogeneous_dataset
    df, true_cates, true_ate = make_fully_heterogeneous_dataset(n_obs=1000,
                                                                n_confounders=5,
                                                                theta=4.0,
                                                                seed=1)

    print(f"True CATEs: {true_cates[:5]}")
    print(f"True ATE: {true_ate}")
    print(df.head())
    ```
    """

    np.random.seed(seed)

    v = np.random.uniform(
        size=[
            n_obs,
        ]
    )
    zeta = np.random.standard_normal(
        size=[
            n_obs,
        ]
    )

    cov_mat = toeplitz([np.power(0.5, k) for k in range(n_confounders)])
    x = np.random.multivariate_normal(
        np.zeros(n_confounders),
        cov_mat,
        size=[
            n_obs,
        ],
    )

    R2_y = doubleml_kwargs.get("R2_y", 0.5)
    R2_d = doubleml_kwargs.get("R2_d", 0.5)

    beta = [1 / (k**2) for k in range(1, n_confounders + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_y = np.sqrt(R2_y / ((1 - R2_y) * b_sigma_b))
    c_d = np.sqrt(np.pi**2 / 3.0 * R2_d / ((1 - R2_d) * b_sigma_b))

    xx = np.exp(np.dot(x, np.multiply(beta, c_d)))
    d = 1.0 * ((xx / (1 + xx)) > v)

    def y_func(d, x, theta):
        return d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta

    y = y_func(d, x, theta)

    x_cols = [f"X{i + 1}" for i in np.arange(n_confounders)]
    df = pd.DataFrame(np.column_stack((x, y, d)), columns=x_cols + ["y", "d"])

    d1 = np.ones_like(d)
    d0 = np.zeros_like(d)

    true_cates = y_func(d1, x, theta) - y_func(d0, x, theta)
    true_ate = true_cates.mean()

    return df, true_cates, true_ate


# TODO: Open PR w/ DoWhy to add this functionality to the library
def make_dowhy_linear_dataset(
    beta: float = 2.0,
    n_obs: int = 1000,
    n_confounders: int = 10,
    n_discrete_confounders: int = 0,
    n_effect_modifiers: int = 5,
    n_discrete_effect_modifiers: int = 0,
    n_treatments: int = 1,
    binary_treatment: bool = False,
    categorical_treatment: bool = False,
    binary_outcome: bool = False,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    """
    Simulate a linear data generating process with flexible configurations. The outcome and treatment can take on different data types.
    The dataset is generated using a modified version of the `make_linear_data` function from the [`dowhy` package](https://www.pywhy.org/dowhy/v0.11.1/index.html).

    The general form of the data generating process is:

    $$
    y_i = \\tau (\mathbf{X_i}) \mathbf{D_i} + g(\mathbf{W_i}) + \epsilon_i
    $$
    $$
    \mathbf{D_i}=f(\mathbf{W_i})+\eta_i
    $$

    where $y_i$ is the outcome, $\mathbf{D_i}$ are the treatment(s), $\mathbf{X_i}$ are the effect modifiers (utilized for effect heterogeneity only), $\mathbf{W_i}$ are the confounders, $\epsilon_i$ and $\eta_i$ are the error terms,
    $\\tau$ is the linear CATE function, $g$ is the linear outcome function, and $f$ is the linear treatment function.

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        X((X))-->Y((Y));
        W((W))-->Y((Y));
        W((W))-->D((D));
        D((D))-->|"τ(X)"|Y((Y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    beta : float
        The base effect size of the treatment. Note, this differs from the ATE with effect modifiers.
    n_obs : int
        The number of observations to generate.
    n_confounders : int
        The number of confounders $\mathbf{W_i}$ to generate.
    n_discrete_confounders : int
        The number of discrete confounders to generate.
    n_effect_modifiers : int
        The number of effect modifiers $\mathbf{X_i}$ to generate.
    n_discrete_effect_modifiers : int
        The number of discrete effect modifiers to generate.
    n_treatments : int
        The number of treatments $\mathbf{D_i}$ to generate.
    binary_treatment : bool
        Whether the treatment is binary or continuous.
    categorical_treatment : bool
        Whether the treatment is categorical or continuous.
    binary_outcome : bool
        Whether the outcome is binary or continuous.
    seed : int | None
        The seed to use for the random number generator.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset where y is the outcome, d are the treatment(s), X are the covariates that are utilized for heterogeneity only, and W are the confounders.
    true_cates : dict[str,np.ndarray]
        The true conditional average treatment effects for each treatment.
    true_ate : dict[str, float]
        The true average treatment effect for each treatment.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import make_dowhy_linear_dataset
    df, true_cates, true_ate = make_dowhy_linear_dataset(beta=2.0,
                                                        n_obs=1000,
                                                        n_confounders=10,
                                                        n_discrete_confounders=0,
                                                        n_effect_modifiers=5,
                                                        n_discrete_effect_modifiers=0,
                                                        n_treatments=1,
                                                        binary_treatment=False,
                                                        categorical_treatment=False,
                                                        binary_outcome=False,
                                                        seed=1)

    print(f"True CATEs: {true_cates['d1'][:5]}")
    print(f"True ATE: {true_ate}")
    print(df.head())
    ```
    """
    np.random.seed(seed)

    # Mapping parameters to the dowhy format
    beta = beta
    num_samples = n_obs
    num_common_causes = n_confounders
    num_discrete_common_causes = n_discrete_confounders
    num_effect_modifiers = n_effect_modifiers
    num_discrete_effect_modifiers = n_discrete_effect_modifiers
    num_treatments = n_treatments
    treatment_is_binary = binary_treatment
    treatment_is_category = categorical_treatment
    outcome_is_binary = binary_outcome
    num_instruments = 0
    # num_discrete_instruments = 0
    num_frontdoor_variables = 0
    one_hot_encode = False
    stddev_treatment_noise = 1
    stddev_outcome_noise = 0.1
    stochastic_discretization = True

    assert not (treatment_is_binary and treatment_is_category)
    W, X, Z, FD, c1, c2, ce, cz, cfd1, cfd2 = [None] * 10
    W_with_dummy, X_with_categorical = (None, None)
    beta = np.array(beta, dtype=np.float64, ndmin=1)
    if num_treatments is None:
        num_treatments = beta.size
    else:
        beta = np.resize(beta, num_treatments)
    num_cont_common_causes = num_common_causes - num_discrete_common_causes
    num_cont_effect_modifiers = num_effect_modifiers - num_discrete_effect_modifiers
    if num_common_causes > 0:
        range_c1 = 0.5 + np.max(np.absolute(beta)) * 0.5
        range_c2 = 0.5 + np.max(np.absolute(beta)) * 0.5
        means = np.random.uniform(-1, 1, num_common_causes)
        cov_mat = np.diag(np.ones(num_common_causes))
        W = np.random.multivariate_normal(means, cov_mat, num_samples)
        W_with_dummy = convert_to_categorical(
            W,
            num_common_causes,
            num_discrete_common_causes,
            quantiles=[0.25, 0.5, 0.75],
            one_hot_encode=one_hot_encode,
        )
        c1 = np.random.uniform(0, range_c1, (W_with_dummy.shape[1], num_treatments))
        c2 = np.random.uniform(0, range_c2, W_with_dummy.shape[1])

    if num_instruments > 0:
        range_cz = 1 + np.max(np.absolute(beta))
        p = np.random.uniform(0, 1, num_instruments)
        Z = np.zeros((num_samples, num_instruments))
        for i in range(num_instruments):
            if (i % 2) == 0:
                Z[:, i] = np.random.binomial(n=1, p=p[i], size=num_samples)
            else:
                Z[:, i] = np.random.uniform(0, 1, size=num_samples)
        # TODO Ensure that we do not generate weak instruments
        cz = np.random.uniform(
            range_cz - (range_cz * 0.05),
            range_cz + (range_cz * 0.05),
            (num_instruments, num_treatments),
        )
    if num_effect_modifiers > 0:
        range_ce = 0.5 + np.max(np.absolute(beta)) * 0.5
        means = np.random.uniform(-1, 1, num_effect_modifiers)
        cov_mat = np.diag(np.ones(num_effect_modifiers))
        X = np.random.multivariate_normal(means, cov_mat, num_samples)
        X_with_categorical = convert_to_categorical(
            X,
            num_effect_modifiers,
            num_discrete_effect_modifiers,
            quantiles=[0.25, 0.5, 0.75],
            one_hot_encode=one_hot_encode,
        )
        ce = np.random.uniform(0, range_ce, X_with_categorical.shape[1])
    # TODO - test all our methods with random noise added to covariates (instead of the stochastic treatment assignment)

    t = np.random.normal(0, stddev_treatment_noise, (num_samples, num_treatments))
    if num_common_causes > 0:
        t += W_with_dummy @ c1  # + np.random.normal(0, 0.01)
    if num_instruments > 0:
        t += Z @ cz
    # Converting treatment to binary if required
    if treatment_is_binary:
        t = np.vectorize(convert_to_binary)(t)
    elif treatment_is_category:
        t = np.vectorize(stochastically_convert_to_three_level_categorical)(t)

    # Generating frontdoor variables if asked for
    if num_frontdoor_variables > 0:
        range_cfd1 = np.max(np.absolute(beta)) * 0.5
        range_cfd2 = np.max(np.absolute(beta)) * 0.5
        cfd1 = np.random.uniform(
            0, range_cfd1, (num_treatments, num_frontdoor_variables)
        )
        cfd2 = np.random.uniform(0, range_cfd2, num_frontdoor_variables)
        FD_noise = np.random.normal(0, 1, (num_samples, num_frontdoor_variables))
        FD = FD_noise
        FD += t @ cfd1
        if num_common_causes > 0:
            range_c1_frontdoor = range_c1 / 10.0
            c1_frontdoor = np.random.uniform(
                0, range_c1_frontdoor, (W_with_dummy.shape[1], num_frontdoor_variables)
            )
            FD += W_with_dummy @ c1_frontdoor

    def _compute_y(t, W, X, FD, beta, c2, ce, cfd2, stddev_outcome_noise, seed=None):
        np.random.seed(seed)
        y = np.random.normal(0, stddev_outcome_noise, num_samples)
        if num_frontdoor_variables > 0:
            y += FD @ cfd2
        else:
            # NOTE: We are assuming a linear relationship *even when t is categorical* and integer coded.
            # For categorical t, this example dataset has the effect size for category 2 being exactly
            # double the effect for category 1
            # This could be changed at this stage by one-hot encoding t and using a custom beta that
            # sets a different effect for each category {0, 1, 2}
            y += t @ beta
        if num_common_causes > 0:
            y += W @ c2
        if num_effect_modifiers > 0:
            y += (X @ ce) * np.prod(t, axis=1)
        if outcome_is_binary:
            y = np.vectorize(convert_to_binary)(y, stochastic_discretization)
        return y

    y = _compute_y(
        t,
        W_with_dummy,
        X_with_categorical,
        FD,
        beta,
        c2,
        ce,
        cfd2,
        stddev_outcome_noise,
        seed=seed,
    )

    data = np.column_stack((t, y))
    if num_common_causes > 0:
        data = np.column_stack((W_with_dummy, data))
    if num_instruments > 0:
        data = np.column_stack((Z, data))
    if num_effect_modifiers > 0:
        data = np.column_stack((X_with_categorical, data))
    if num_frontdoor_variables > 0:
        data = np.column_stack((FD, data))

    # Computing ATE
    ates = {}
    cates = {}
    FD_T1, FD_T0 = None, None
    for i in range(num_treatments):
        T1 = np.copy(t)
        T0 = np.copy(t)

        T1[:, i] = 1
        T0[:, i] = 0

        if num_frontdoor_variables > 0:
            FD_T1 = FD_noise + (T1 @ cfd1)
            FD_T0 = FD_noise + (T0 @ cfd1)

        ate = np.mean(
            _compute_y(
                T1,
                W_with_dummy,
                X_with_categorical,
                FD_T1,
                beta,
                c2,
                ce,
                cfd2,
                stddev_outcome_noise,
                seed=seed,
            )
            - _compute_y(
                T0,
                W_with_dummy,
                X_with_categorical,
                FD_T0,
                beta,
                c2,
                ce,
                cfd2,
                stddev_outcome_noise,
                seed=seed,
            )
        )

        cate = _compute_y(
            T1,
            W_with_dummy,
            X_with_categorical,
            FD_T1,
            beta,
            c2,
            ce,
            cfd2,
            stddev_outcome_noise,
            seed=seed,
        ) - _compute_y(
            T0,
            W_with_dummy,
            X_with_categorical,
            FD_T0,
            beta,
            c2,
            ce,
            cfd2,
            stddev_outcome_noise,
            seed=seed,
        )

        if treatment_is_category:
            T2 = np.copy(t)
            T2[:, i] = 2

            ate2 = np.mean(
                _compute_y(
                    T2,
                    W_with_dummy,
                    X_with_categorical,
                    FD_T1,
                    beta,
                    c2,
                    ce,
                    cfd2,
                    stddev_outcome_noise,
                    seed=seed,
                )
                - _compute_y(
                    T0,
                    W_with_dummy,
                    X_with_categorical,
                    FD_T0,
                    beta,
                    c2,
                    ce,
                    cfd2,
                    stddev_outcome_noise,
                    seed=seed,
                )
            )

            cate2 = _compute_y(
                T2,
                W_with_dummy,
                X_with_categorical,
                FD_T1,
                beta,
                c2,
                ce,
                cfd2,
                stddev_outcome_noise,
                seed=seed,
            ) - _compute_y(
                T0,
                W_with_dummy,
                X_with_categorical,
                FD_T0,
                beta,
                c2,
                ce,
                cfd2,
                stddev_outcome_noise,
                seed=seed,
            )

            ates[f"d{i+1}"] = [ate, ate2]
            cates[f"d{i+1}"] = [cate, cate2]
        else:
            ates[f"d{i+1}"] = ate
            cates[f"d{i+1}"] = cate

    treatments = [("d" + str(i)) for i in range(1, num_treatments + 1)]
    outcome = "y"
    # constructing column names for one-hot encoded discrete features
    common_causes = construct_col_names(
        "W",
        num_common_causes,
        num_discrete_common_causes,
        num_discrete_levels=4,
        one_hot_encode=one_hot_encode,
    )
    instruments = [("Z" + str(i)) for i in range(0, num_instruments)]
    frontdoor_variables = [("FD" + str(i)) for i in range(0, num_frontdoor_variables)]
    effect_modifiers = construct_col_names(
        "X",
        num_effect_modifiers,
        num_discrete_effect_modifiers,
        num_discrete_levels=4,
        one_hot_encode=one_hot_encode,
    )
    col_names = (
        frontdoor_variables
        + effect_modifiers
        + instruments
        + common_causes
        + treatments
        + [outcome]
    )
    data = pd.DataFrame(data, columns=col_names)
    # Specifying the correct dtypes
    if treatment_is_binary:
        data = data.astype({tname: "bool" for tname in treatments}, copy=False)
    elif treatment_is_category:
        data = data.astype({tname: "category" for tname in treatments}, copy=False)
    if outcome_is_binary:
        data = data.astype({outcome: "bool"}, copy=False)
    if num_discrete_common_causes > 0 and not one_hot_encode:
        data = data.astype(
            {wname: "int64" for wname in common_causes[num_cont_common_causes:]},
            copy=False,
        )
        data = data.astype(
            {wname: "category" for wname in common_causes[num_cont_common_causes:]},
            copy=False,
        )
    if num_discrete_effect_modifiers > 0 and not one_hot_encode:
        data = data.astype(
            {
                emodname: "int64"
                for emodname in effect_modifiers[num_cont_effect_modifiers:]
            },
            copy=False,
        )
        data = data.astype(
            {
                emodname: "category"
                for emodname in effect_modifiers[num_cont_effect_modifiers:]
            },
            copy=False,
        )

    df = data
    true_ate = ates
    true_cates = cates

    return df, true_cates, true_ate


# Plucked from DoWhy
def construct_col_names(
    name, num_vars, num_discrete_vars, num_discrete_levels, one_hot_encode
):
    colnames = [(name + str(i)) for i in range(0, num_vars - num_discrete_vars)]
    if one_hot_encode:
        discrete_colnames = [
            name + str(i) + "_" + str(j)
            for i in range(num_vars - num_discrete_vars, num_vars)
            for j in range(0, num_discrete_levels)
        ]
        colnames = colnames + discrete_colnames
    else:
        colnames = colnames + [
            (name + str(i)) for i in range(num_vars - num_discrete_vars, num_vars)
        ]

    return colnames


def convert_to_binary(x, stochastic=True):
    p = sigmoid(x)
    if stochastic:
        return choice([0, 1], 1, p=[1 - p, p])
    else:
        return int(p > 0.5)


def stochastically_convert_to_three_level_categorical(x):
    p = sigmoid(x)
    return choice([0, 1, 2], 1, p=[0.8 * (1 - p), 0.8 * p, 0.2])


def convert_to_categorical(
    arr, num_vars, num_discrete_vars, quantiles=[0.25, 0.5, 0.75], one_hot_encode=False
):
    arr_with_dummy = arr.copy()
    # Below loop assumes that the last indices of W are alwawys converted to discrete
    for arr_index in range(num_vars - num_discrete_vars, num_vars):
        # one-hot encode discrete W
        arr_bins = np.quantile(arr[:, arr_index], q=quantiles)
        arr_categorical = np.digitize(arr[:, arr_index], bins=arr_bins)
        if one_hot_encode:
            dummy_vecs = np.eye(len(quantiles) + 1)[arr_categorical]
            arr_with_dummy = np.concatenate((arr_with_dummy, dummy_vecs), axis=1)
        else:
            arr_with_dummy = np.concatenate(
                (arr_with_dummy, arr_categorical[:, np.newaxis]), axis=1
            )
    # Now deleting the old continuous value
    for arr_index in range(num_vars - 1, num_vars - num_discrete_vars - 1, -1):
        arr_with_dummy = np.delete(arr_with_dummy, arr_index, axis=1)
    return arr_with_dummy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
