import re

import numpy as np
import pandas as pd
import pytest
from typeguard import suppress_type_checks
from typing_extensions import Callable

from caml.extensions.synthetic_data import SyntheticDataGenerator

pytestmark = [pytest.mark.extensions, pytest.mark.synthetic_data]


class TestSyntheticDataGeneratorIntegration:
    @suppress_type_checks
    @pytest.mark.parametrize(
        (
            "n_obs,n_cont_outcomes,n_binary_outcomes,n_cont_treatments,n_binary_treatments,n_discrete_treatments,causal_model_functional_form,n_features"
        ),
        [
            (1000, 1, 1, 1, 1, 1, "linear", 4),
            (1000, 1, 1, 1, 1, 1, "nonlinear", 4),
            (1000, 1, 1, 1, 0, 0, "linear", 4),
            (1000, 1, 1, 1, 0, 0, "nonlinear", 4),
            (1000, 1, 1, 0, 1, 0, "linear", 4),
            (1000, 1, 1, 0, 1, 0, "nonlinear", 4),
            (1000, 1, 1, 0, 0, 1, "linear", 4),
            (1000, 1, 1, 0, 0, 1, "nonlinear", 4),
            (1000, 1, 1, 1, 1, 1, "linear", 0),
            (1000, 1, 1, 1, 1, 1, "nonlinear", 0),
            (1000, 1, 1, 1, 1, 1, "bad", 4),  # Fail on bad functional form
            (1000, 0, 0, 1, 1, 1, "linear", 4),  # Fail on no outcomes
            (1000, 1, 1, 0, 0, 0, "linear", 4),  # Fail on no treatments
            (0, 1, 1, 1, 1, 1, "linear", 4),  # Fail on no observations
        ],
        ids=[
            "linear_all_treatment_types",
            "nonlinear_all_treatment_types",
            "linear_only_cont_treatments",
            "nonlinear_only_cont_treatments",
            "linear_only_binary_treatments",
            "nonlinear_only_binary_treatments",
            "linear_only_discrete_treatments",
            "nonlinear_only_discrete_treatments",
            "linear_no_covariates",
            "nonlinear_no_covariates",
            "bad_functional_form",
            "no_outcomes",
            "no_treatments",
            "no_observations",
        ],
    )
    def test_init(
        self,
        n_obs,
        n_cont_outcomes,
        n_binary_outcomes,
        n_cont_treatments,
        n_binary_treatments,
        n_discrete_treatments,
        causal_model_functional_form,
        n_features,
    ):
        n_nonlinear_transformations = 5

        def call():
            return SyntheticDataGenerator(
                n_obs=n_obs,
                n_cont_outcomes=n_cont_outcomes,
                n_binary_outcomes=n_binary_outcomes,
                n_cont_treatments=n_cont_treatments,
                n_binary_treatments=n_binary_treatments,
                n_discrete_treatments=n_discrete_treatments,
                n_cont_confounders=n_features,
                n_binary_confounders=n_features,
                n_discrete_confounders=n_features,
                n_cont_modifiers=n_features,
                n_binary_modifiers=n_features,
                n_discrete_modifiers=n_features,
                n_confounding_modifiers=n_features,
                causal_model_functional_form=causal_model_functional_form,
                n_nonlinear_transformations=n_nonlinear_transformations,
            )

        if (
            causal_model_functional_form == "bad"
            or n_cont_outcomes + n_binary_outcomes == 0
            or n_cont_treatments + n_binary_treatments + n_discrete_treatments == 0
            or n_obs == 0
        ):
            with pytest.raises(ValueError):
                call()
        else:
            gen = call()

            assert (
                gen._n_nonlinear_transformations == n_nonlinear_transformations
                if causal_model_functional_form == "nonlinear"
                else gen._n_nonlinear_transformations is None
            )

            assert gen.df.shape == (
                n_obs,
                n_cont_outcomes
                + n_binary_outcomes
                + n_cont_treatments
                + n_binary_treatments
                + n_discrete_treatments
                + n_features * 6,
            )

            sum_categories = 0
            for dis_t in [c for c in gen.df if re.match(r"T[0-9]+", c) and "dis" in c]:
                sum_categories += len(gen.df[dis_t].unique())

            n_treatment_effects = (n_cont_outcomes + n_binary_outcomes) * (
                n_cont_treatments
                + n_binary_treatments
                + sum_categories
                - n_discrete_treatments  # Subtract reference group
            )
            assert gen.cates.shape == (n_obs, n_treatment_effects)
            assert gen.ates.shape == (n_treatment_effects, 2)
            assert np.allclose(gen.cates.mean(axis=0), gen.ates["ATE"])

            for dep_var in [
                c
                for c in gen.df.columns
                if re.match(r"Y[0-9]+", c) or re.match(r"T[0-9]+", c)
            ]:
                dep_var_dgp = gen.dgp[dep_var]
                params = dep_var_dgp["params"]
                formula = dep_var_dgp["formula"]
                noise = dep_var_dgp["noise"]
                raw_scores = dep_var_dgp["raw_scores"]
                function = dep_var_dgp["function"]

                assert isinstance(dep_var_dgp, dict)
                assert isinstance(formula, str)
                assert isinstance(params, np.ndarray)
                assert isinstance(noise, np.ndarray)
                assert isinstance(raw_scores, np.ndarray)
                assert isinstance(function, Callable)

                if formula != "":
                    design_matrix = gen.create_design_matrix(
                        gen.df, formula=formula, return_type="dataframe"
                    )
                    assert isinstance(design_matrix, pd.DataFrame)

                    # Recreate dep_var
                    assert np.allclose(
                        raw_scores, function(design_matrix, params, noise)
                    )

                    # Test treatment effect estimation
                    if "Y" in dep_var:
                        for t in [c for c in gen.df.columns if c.startswith("T")]:
                            data_treat = gen.df.copy()
                            data_cont = gen.df.copy()
                            if "bin" in t:
                                data_treat[t] = 1
                                data_cont[t] = 0
                            elif "cont" in t:
                                data_treat[t] = data_treat[t] + 1
                            else:
                                continue

                            dm1 = gen.create_design_matrix(
                                data_treat, formula=formula, return_type="dataframe"
                            )
                            dm0 = gen.create_design_matrix(
                                data_cont, formula=formula, return_type="dataframe"
                            )

                            treat_col = [
                                c for c in gen.cates.columns if t in c and dep_var in c
                            ][0]
                            assert np.allclose(
                                gen.cates[treat_col],
                                function(dm1, params, np.zeros_like(noise))
                                - function(dm0, params, np.zeros_like(noise)),
                            )
