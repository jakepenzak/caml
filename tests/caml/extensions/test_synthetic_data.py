import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from typing_extensions import Callable

from caml.extensions.synthetic_data import (
    SyntheticDataGenerator,
    _truncate_and_renormalize_probabilities,
    make_fully_heterogeneous_dataset,
    make_partially_linear_dataset_constant,
    make_partially_linear_dataset_simple,
)

pytestmark = [pytest.mark.extensions, pytest.mark.synthetic_data]


class TestSyntheticDataGenerator:
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
    def test_init(  # Testing everything in one pass, yuck! Not core class.
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
                assert isinstance(formula, str | None)
                assert isinstance(params, np.ndarray)
                assert isinstance(noise, np.ndarray)
                assert isinstance(raw_scores, np.ndarray)
                assert isinstance(function, Callable)

                if formula is not None:
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


class TestFunctionals:
    @pytest.mark.parametrize(
        ("probs, expected"),
        [
            (
                [0, 0.05, 0.5, 0.95, 1],
                [0.1, 0.1, 0.5, 0.9, 0.9],
            ),
            (
                [
                    [0, 0.05, 0.95, 1],
                    [0.8, 0.75, 0.05, 0],
                    [0.2, 0.2, 0, 0],
                ],
                [
                    [0.09, 0.10, 0.82, 0.82],
                    [0.73, 0.71, 0.09, 0.09],
                    [0.18, 0.19, 0.09, 0.09],
                ],
            ),
        ],
    )
    def test__truncate_and_renormalize_probabilities(self, probs, expected):
        truncated_probs = _truncate_and_renormalize_probabilities(
            np.array(probs).T, epsilon=0.1
        )
        assert_allclose(truncated_probs, np.array(expected).T, atol=0.01)

    @pytest.mark.parametrize("n_obs", [1000, 10000])
    @pytest.mark.parametrize("n_confounders", [5, 10])
    @pytest.mark.parametrize("dim_heterogeneity", [1, 2, 3])
    @pytest.mark.parametrize("binary_treatment", [True, False])
    def test_make_partially_linear_dataset_simple(
        self,
        n_obs,
        n_confounders,
        dim_heterogeneity,
        binary_treatment,
    ):
        if dim_heterogeneity == 3:
            with pytest.raises(ValueError):
                make_partially_linear_dataset_simple(
                    dim_heterogeneity=dim_heterogeneity
                )
        else:
            df, cates, ate = make_partially_linear_dataset_simple(
                n_obs=n_obs,
                n_confounders=n_confounders,
                dim_heterogeneity=dim_heterogeneity,
                binary_treatment=binary_treatment,
            )
            assert df.shape == (n_obs, n_confounders + 2)
            assert cates.shape == (n_obs,)
            assert isinstance(ate, float)
            if binary_treatment:
                assert df["d"].unique().shape[0] == 2
            else:
                assert df["d"].unique().shape[0] != 2

            assert ate == pytest.approx(4.5, abs=0.5)

    @pytest.mark.parametrize("n_obs", [1000, 10000])
    @pytest.mark.parametrize("exp_ate", [4.5, 15.5])
    @pytest.mark.parametrize("n_confounders", [5, 15])
    @pytest.mark.parametrize(
        "dgp", ["make_plr_CCDDHNR2018", "make_plr_turrell2018", "bad"]
    )
    def test_make_partially_linear_dataset_constant(
        self,
        n_obs,
        exp_ate,
        n_confounders,
        dgp,
    ):
        if dgp == "bad":
            with pytest.raises(ValueError):
                make_partially_linear_dataset_constant(dgp=dgp)
        else:
            df, cates, ate = make_partially_linear_dataset_constant(
                n_obs=n_obs,
                ate=exp_ate,
                n_confounders=n_confounders,
                dgp=dgp,
            )
            assert df.shape == (n_obs, n_confounders + 2)
            assert cates.shape == (n_obs,)
            assert isinstance(ate, float)
            assert ate == exp_ate
            assert np.all(cates == exp_ate)

    @pytest.mark.parametrize("n_obs", [1000, 10000])
    @pytest.mark.parametrize("n_confounders", [5, 15])
    @pytest.mark.parametrize("theta", [4.0, 5.6])
    def test_make_fully_heterogeneous_dataset(self, n_obs, n_confounders, theta):
        df, cates, ate = make_fully_heterogeneous_dataset(
            n_obs=n_obs,
            n_confounders=n_confounders,
            theta=theta,
        )
        assert df.shape == (n_obs, n_confounders + 2)
        assert cates.shape == (n_obs,)
        assert isinstance(ate, float)
        assert ate == pytest.approx(theta, abs=0.2)
