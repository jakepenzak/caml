import importlib
import sys
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import mean_squared_error
from statsmodels.formula.api import ols

import caml.core.ols as ols_mod
from caml import FastOLS

pytestmark = [pytest.mark.core, pytest.mark.ols]
N = 1000


@pytest.fixture(params=["jax", "numpy"], ids=["with_jax", "with_numpy"])
def backend(request, monkeypatch):
    """Simulate two backends.

    - 'jax'   → jax, jax.numpy, jax.scipy.stats, _HAS_JAX=True
    - 'numpy' → numpy, scipy.stats, _HAS_JAX=False.
    """
    if request.param == "jax":
        import jax
        import jax.numpy as jnp
        import jax.scipy.stats as jstats

        monkeypatch.setattr(ols_mod, "_HAS_JAX", True)
        monkeypatch.setattr(ols_mod, "jax", jax)
        monkeypatch.setattr(ols_mod, "jnp", jnp)
        monkeypatch.setattr(ols_mod, "jstats", jstats)
    else:
        import numpy as np
        import scipy.stats as sstats

        monkeypatch.setattr(ols_mod, "_HAS_JAX", False)
        monkeypatch.setattr(ols_mod, "jax", None)
        monkeypatch.setattr(ols_mod, "jnp", np)
        monkeypatch.setattr(ols_mod, "jstats", sstats)

    return request.param


@pytest.fixture
def dgp():
    """Generate multi-dimensional outcome dgp."""
    rng = np.random.default_rng(123)

    # binary treatment
    T = rng.integers(0, 2, size=N)
    # covariates
    X1 = rng.normal(size=N)
    X2 = rng.normal(size=N) * 2
    W = rng.normal(size=N)
    # Groups
    G1 = rng.integers(1, 4, size=N)
    G2 = rng.integers(1, 4, size=N)

    # outcomes with known linear relationships
    y1_params = [1.5, 0.5, 0.2, -0.2, -0.5, 0.2, 1.8, 0.5, -0.5, 0.1]

    def Y1(T, X1, X2, G1, G2, W):
        y1 = y1_params @ np.array(
            [T, X1, X1 * T, X2, X2 * T, G1, G1 * T, G2, G2 * T, W]
        ) + rng.normal(scale=0.1, size=N)
        return y1

    y2_params = [-0.7, 0.3, 1.2, 0.8, -0.4, 0.5, -3, 0.5, 4, 0.2]

    def Y2(T, X1, X2, G1, G2, W):
        y2 = y2_params @ np.array(
            [T, X1, X1 * T, X2, X2 * T, G1, G1 * T, G2, G2 * T, W]
        ) + rng.normal(scale=0.1, size=N)
        return y2

    CATE_Y1 = Y1(np.ones(N), X1, X2, G1, G2, W) - Y1(np.zeros(N), X1, X2, G1, G2, W)
    CATE_Y2 = Y2(np.ones(N), X1, X2, G1, G2, W) - Y2(np.zeros(N), X1, X2, G1, G2, W)

    data = {
        "T": T,
        "X1": X1,
        "X2": X2,
        "W": W,
        "G1": G1,
        "G2": G2,
        "Y1": Y1(T, X1, X2, G1, G2, W),
        "Y2": Y2(T, X1, X2, G1, G2, W),
    }

    effects = {
        "CATE_Y1": CATE_Y1,
        "CATE_Y2": CATE_Y2,
    }
    return {"df": data, "effects": effects}


@pytest.fixture
def pd_df(dgp):
    return pd.DataFrame(dgp["df"])


@pytest.fixture
def pl_df(dgp):
    return pl.DataFrame(dgp["df"])


@pytest.fixture
def ps_df(dgp, request):
    """PySpark DataFrame fixture that skips if spark session fails to create."""
    try:
        spark = request.getfixturevalue("spark")
        return spark.createDataFrame(pd.DataFrame(dgp["df"]))
    except Exception as e:
        pytest.skip(f"Skipping PySpark test due to error: {str(e)}")


@pytest.fixture
def fo_obj(dgp):
    return FastOLS(
        Y=[c for c in dgp["df"].keys() if "Y" in c],
        T="T",
        G=[c for c in dgp["df"].keys() if "G" in c],
        X=[c for c in dgp["df"].keys() if "X" in c],
        W=[c for c in dgp["df"].keys() if "W" in c],
        discrete_treatment=True,
        engine="cpu",
    )


@pytest.fixture
def DummyJax():
    class DummyJax:
        @staticmethod
        def devices(kind):
            # Simulate JAX running, but no GPU devices available:
            raise RuntimeError("No GPU found")

    return DummyJax()


class TestFastOLSInitialization:
    @pytest.mark.parametrize(
        "discrete_treatment",
        [True, False],
        ids=["Discrete", "Continuous"],
    )
    def test_valid_instantiation_sets_attributes(self, discrete_treatment):
        fo = FastOLS(
            Y=["Y1", "Y2"],
            T="T",
            G=["G1", "G2"],
            X=["X1"],
            W=["W1"],
            discrete_treatment=discrete_treatment,
            engine="cpu",
        )
        assert fo.Y == ["Y1", "Y2"]
        assert fo.T == "T"
        assert fo.G == ["G1", "G2"]
        assert fo.X == ["X1"]
        assert fo.W == ["W1"]
        assert fo.discrete_treatment is discrete_treatment
        assert fo.engine == "cpu"
        assert fo._fitted is False
        assert fo.results == {}
        if discrete_treatment:
            assert (
                fo.formula.replace(" ", "")
                == "Y1+Y2~C(T)+C(G1)*C(T)+C(G2)*C(T)+X1*C(T)+W1"
            )
        else:
            assert fo.formula.replace(" ", "") == "Y1+Y2~T+C(G1)*T+C(G2)*T+X1*T+W1"

        summary = (
            "================== FastOLS Object ==================\n"
            + f"Engine: {fo.engine}\n"
            + f"Outcome Variable: {fo.Y}\n"
            + f"Treatment Variable: {fo.T}\n"
            + f"Discrete Treatment: {fo.discrete_treatment}\n"
            + f"Group Variables: {fo.G}\n"
            + f"Features/Confounders for Heterogeneity (X): {fo.X}\n"
            + f"Features/Confounders as Controls (W): {fo.W}\n"
            + f"Formula: {fo.formula}\n"
        )
        assert str(fo) == summary

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError):
            FastOLS(Y=["Y"], T="T", engine="tpu")

    def test_gpu_without_jax_raises(self, backend):
        if not ols_mod._HAS_JAX:
            with pytest.raises(ValueError):
                FastOLS(Y=["Y"], T="T", engine="gpu")

    def test_gpu_fallback_to_cpu(self, monkeypatch, backend, DummyJax):
        if ols_mod._HAS_JAX:
            monkeypatch.setattr(ols_mod, "jax", DummyJax)
            fo = FastOLS(Y=["Y"], T="T", engine="gpu")
            assert fo.engine == "cpu"


@pytest.mark.parametrize(
    "df_fixture",
    ["pd_df", "pl_df", "ps_df", "invalid_df"],
    ids=["Pandas", "Polars", "PySpark", "Invalid"],
)
def test__convert_dataframe_to_pandas(df_fixture, request):
    """Test conversion of different DataFrame types to pandas."""
    try:
        if df_fixture == "invalid_df":
            df_fxt = {"A": [1, 2, 3]}
        else:
            df_fxt = request.getfixturevalue(df_fixture)
    except pytest.FixtureLookupError as e:
        df_fxt = None
        pytest.skip(f"Skipping test with {df_fixture} due to fixture error: {str(e)}")

    fo_obj = FastOLS(Y=["Y"], T="T")

    if df_fixture == "invalid_df":
        with pytest.raises(ValueError):
            fo_obj._convert_dataframe_to_pandas(df_fxt, groups=["G1", "G2"])
    else:
        df = fo_obj._convert_dataframe_to_pandas(df_fxt, groups=["G1", "G2"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (N, len(df_fxt.columns))
        assert sorted(df.columns) == sorted(df_fxt.columns)
        assert df["G1"].dtype == "category"
        assert df["G2"].dtype == "category"
        assert df["T"].dtype == "int64"
        assert df["W"].dtype == "float64"
        assert df["X1"].dtype == "float64"
        assert df["X2"].dtype == "float64"
        assert df["Y1"].dtype == "float64"
        assert df["Y2"].dtype == "float64"


class TestFastOLSFittingAndEstimation:
    @pytest.mark.parametrize("robust_vcv", [True, False], ids=["Robust", "Non-Robust"])
    @pytest.mark.parametrize(
        "estimate_effects", [True, False], ids=["Effects", "No Effects"]
    )
    def test_fit(self, fo_obj, pd_df, robust_vcv, estimate_effects):
        """Test fit method using statsmodels ols as benchmark."""
        fo_obj.fit(pd_df, estimate_effects=estimate_effects, robust_vcv=robust_vcv)
        assert fo_obj._fitted

        for k in ["params", "vcv", "std_err", "treatment_effects"]:
            assert k in fo_obj.results

        for i, y in enumerate([c for c in pd_df.columns if "Y" in c]):
            statsmod = ols(formula=f"{y} ~ {fo_obj.formula.split('~')[1]}", data=pd_df)

            if robust_vcv:
                statsmod = statsmod.fit(cov_type="HC0")
            else:
                statsmod = statsmod.fit()

            assert np.allclose(fo_obj.results["params"][:, i], statsmod.params)
            assert np.allclose(fo_obj.results["vcv"][i, :, :], statsmod.cov_params())
            assert np.allclose(fo_obj.results["std_err"][:, i], statsmod.bse)

        # Make sure non-binary discrete treatments throw an error (not supported yet)
        random_indices = np.random.choice(pd_df.index, size=10, replace=False)
        pd_df.loc[random_indices, "T"] = 3

        with pytest.raises(ValueError):
            fo_obj.fit(pd_df, estimate_effects=estimate_effects, robust_vcv=robust_vcv)

    def test_fit_with_no_groups(self, pd_df, fo_obj):
        # No passed groups will return no group treatment effects
        fo_obj.G = None
        fo_obj.__init__(
            **{
                k: getattr(fo_obj, k)
                for k in ["Y", "T", "G", "X", "W", "discrete_treatment", "engine"]
            }
        )

        fo_obj.fit(pd_df, estimate_effects=True)

        for k, v in fo_obj.results["treatment_effects"].items():
            assert "overall" in k

    @pytest.mark.parametrize(
        "return_results_dict", [True, False], ids=["Results Dict", "No Results Dict"]
    )
    @pytest.mark.parametrize(
        "predict_method", [True, False], ids=["predict", "estimate_cate"]
    )
    def test_estimate_cate(
        self, fo_obj, pd_df, dgp, return_results_dict, predict_method
    ):
        """Test `estimate_cate` and `predict` methods."""
        with pytest.raises(RuntimeError):
            fo_obj.estimate_cate(pd_df, return_results_dict=return_results_dict)

        fo_obj.fit(pd_df, estimate_effects=False)
        if predict_method:
            res = fo_obj.predict(pd_df, return_results_dict=return_results_dict)
        else:
            res = fo_obj.estimate_cate(pd_df, return_results_dict=return_results_dict)

        if return_results_dict:
            assert isinstance(res, dict)
            for k in ["outcome", "cate", "std_err", "t_stat", "pval"]:
                assert k in res
        else:
            assert isinstance(res, jnp.ndarray) or isinstance(res, np.ndarray)

        for i, y in enumerate([c for c in pd_df.columns if "Y" in c]):
            cate_estimated = res["cate"][:, i] if return_results_dict else res[:, i]
            cate_expected = dgp["effects"][f"CATE_{y}"]

            # Check if relative error is small enough (< 10%)
            rel_error = np.mean(
                np.abs((cate_estimated - cate_expected) / cate_expected)
            )
            assert rel_error < 0.1

            # Check small enough Precision in Estimating Heterogenous Treatment Effects (PEHE)
            assert mean_squared_error(cate_estimated, cate_expected) < 0.1

    @pytest.mark.parametrize(
        "return_results_dict", [True, False], ids=["Results Dict", "No Results Dict"]
    )
    def test_estimate_ate(self, fo_obj, pd_df, dgp, return_results_dict):
        """Test `estimate_ate` method."""
        with pytest.raises(RuntimeError):
            fo_obj.estimate_ate(pd_df, return_results_dict=return_results_dict)

        fo_obj.fit(pd_df, estimate_effects=False)
        res = fo_obj.estimate_ate(
            pd_df,
            return_results_dict=return_results_dict,
            group="TestGroup",
            membership="TestMembership",
        )

        if return_results_dict:
            assert isinstance(res, dict)
            for k in [
                "outcome",
                "ate",
                "std_err",
                "t_stat",
                "pval",
                "n",
                "n_treated",
                "n_control",
            ]:
                assert k in res["TestGroup-TestMembership"]
        else:
            assert isinstance(res, jnp.ndarray) or isinstance(res, np.ndarray)

        for i, y in enumerate([c for c in pd_df.columns if "Y" in c]):
            ate_estimated = (
                res["TestGroup-TestMembership"]["ate"][:, i]
                if return_results_dict
                else res[:, i]
            )
            ate_expected = np.mean(dgp["effects"][f"CATE_{y}"])

            assert np.allclose(ate_estimated, ate_expected, atol=0.1)

    @pytest.mark.parametrize(
        "custom_GATE", [True, False], ids=["custom_GATE", "no_custom_GATE"]
    )
    def test_prettify_treatment_effects(self, pd_df, fo_obj, custom_GATE):
        """Test `prettify_treatment_effects` method."""
        fo_obj.fit(pd_df, estimate_effects=True)

        if custom_GATE:
            res = fo_obj.estimate_ate(
                pd_df,
                return_results_dict=True,
            )
            prettified = fo_obj.prettify_treatment_effects(res)
        else:
            prettified = fo_obj.prettify_treatment_effects()
            res = fo_obj.results["treatment_effects"]

        assert isinstance(prettified, pd.DataFrame)

        assert "group" in prettified.columns
        assert "membership" in prettified.columns
        assert "outcome" in prettified.columns
        assert "n" in prettified.columns
        assert "n_treated" in prettified.columns
        assert "n_control" in prettified.columns
        for c in [
            "ate",
            "std_err",
            "t_stat",
            "pval",
        ]:
            assert c in prettified.columns
            # Recurse through dictionary and hstack numpy arrays to compare to prettified column
            stack = None
            for k, v in res.items():
                stack = (
                    np.hstack([stack, v[c].flatten()])
                    if stack is not None
                    else v[c].flatten()
                )

            assert np.allclose(stack, prettified[c])


def test_jax_fallback_to_numpy(monkeypatch):
    """Simulate missing jax and ensure fallback to numpy."""
    # Remove jax from sys.modules if it was loaded
    sys.modules.pop("jax", None)
    sys.modules.pop("jax.numpy", None)
    sys.modules.pop("jax.scipy.stats", None)

    # Patch sys.modules to simulate ImportError
    with mock.patch.dict("sys.modules", {"jax": None}):
        # Reload the module where your fallback logic lives
        import caml.core.ols as ols_mod

        importlib.reload(ols_mod)

        # Check that fallback occurred
        assert not ols_mod._HAS_JAX
        assert ols_mod.jnp.__name__.startswith("numpy")
        assert ols_mod.jstats.__name__.startswith("scipy")
