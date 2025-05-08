import numpy as np
import pytest
from numpy.testing import assert_allclose

from caml.extensions.synthetic_data import (
    _truncate_and_renormalize_probabilities,
    make_partially_linear_dataset_simple,
)

pytestmark = [pytest.mark.extensions, pytest.mark.synthetic_data]


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
    @pytest.mark.parametrize("seed", [None, 1])
    def test_make_partially_linear_dataset_simple(
        self,
        n_obs,
        n_confounders,
        dim_heterogeneity,
        binary_treatment,
        seed,
    ):
        if dim_heterogeneity == 3:
            with pytest.raises(ValueError):
                make_partially_linear_dataset_simple(
                    n_obs=n_obs,
                    n_confounders=n_confounders,
                    dim_heterogeneity=dim_heterogeneity,
                    binary_treatment=binary_treatment,
                    seed=seed,
                )
        else:
            df, cates, ate = make_partially_linear_dataset_simple(
                n_obs=n_obs,
                n_confounders=n_confounders,
                dim_heterogeneity=dim_heterogeneity,
                binary_treatment=binary_treatment,
                seed=seed,
            )
            assert df.shape == (n_obs, n_confounders + 2)
            assert cates.shape == (n_obs,)
            assert isinstance(ate, float)
            if binary_treatment:
                assert df["d"].unique().shape[0] == 2
            else:
                assert df["d"].unique().shape[0] != 2

            if seed == 1:
                assert ate == pytest.approx(4.5, abs=0.6)
