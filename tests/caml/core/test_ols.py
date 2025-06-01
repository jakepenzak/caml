import builtins
import importlib
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols

import caml.core.ols as ols_mod
import caml.generics as gen_mod
from caml import FastOLS

pytestmark = [pytest.mark.core, pytest.mark.ols]
N = 1000


@pytest.fixture(params=["jax", "numpy"], ids=["with_jax", "with_numpy"])
def backend(request, monkeypatch):
    """Drive FastOLS tests under two simulated environments.

    - "with_jax": real JAX is available, so ols_mod._HAS_JAX == True
    - "with_numpy": JAX imports fail, so ols_mod._HAS_JAX == False
    """
    # Save the real __import__ so we can restore it later
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Simulate ImportError for any jax or jax.* import
        if name == "jax" or name.startswith("jax."):
            raise ImportError(f"no module named {name}")
        return real_import(name, globals, locals, fromlist, level)

    if request.param == "jax":
        # Ensure that, if we’d previously reloaded with numpy, we go back to JAX mode
        if not ols_mod._HAS_JAX:
            # Restore real import
            monkeypatch.setattr(builtins, "__import__", real_import)
            importlib.reload(ols_mod)
            importlib.reload(gen_mod)
        # Sanity-check
        assert ols_mod._HAS_JAX
        assert gen_mod._HAS_JAX
    else:
        # Simulate JAX missing
        monkeypatch.setattr(builtins, "__import__", fake_import)
        importlib.reload(ols_mod)
        importlib.reload(gen_mod)
        # Sanity‑check
        assert not gen_mod._HAS_JAX
        assert not ols_mod._HAS_JAX

        import numpy as _np
        import scipy.stats as _sstats

        assert ols_mod.jnp is _np
        assert ols_mod.jstats is _sstats

    yield request.param

    # Teardown: restore import & reload to original module state
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(ols_mod)
    importlib.reload(gen_mod)


@pytest.fixture(params=["cont_T", "binary_T"], ids=["continuous_T", "binary_T"])
def dgp(request):
    """Generate multi-dimensional outcome dgp."""
    rng = np.random.default_rng(123)

    treatment_type = request.param
    if treatment_type == "binary_T":
        # binary treatment
        T = rng.integers(0, 2, size=N)
        col_str = "T_binary"
    else:
        # continuous treatment
        T = rng.normal(size=N)
        col_str = "T_continuous"

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

    if treatment_type == "binary_T":
        CATE_Y1 = Y1(np.ones(N), X1, X2, G1, G2, W) - Y1(np.zeros(N), X1, X2, G1, G2, W)
        CATE_Y2 = Y2(np.ones(N), X1, X2, G1, G2, W) - Y2(np.zeros(N), X1, X2, G1, G2, W)
    else:
        CATE_Y1 = Y1(T + 1, X1, X2, G1, G2, W) - Y1(T, X1, X2, G1, G2, W)
        CATE_Y2 = Y2(T + 1, X1, X2, G1, G2, W) - Y2(T, X1, X2, G1, G2, W)

    data = {
        col_str: T,
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


@pytest.fixture(params=["Pandas", "Polars", "PySpark", "InvalidDF"])
def df_fixture(request, dgp):
    if request.param == "Pandas":
        return pd.DataFrame(dgp["df"])
    elif request.param == "Polars":
        return pl.DataFrame(dgp["df"])
    elif request.param == "PySpark":
        try:
            spark = request.getfixturevalue("spark")
            return spark.createDataFrame(pd.DataFrame(dgp["df"]))
        except Exception as e:
            pytest.skip(f"Skipping PySpark test due to error: {str(e)}")
    elif request.param == "InvalidDF":
        return {"invalid": [1, 2, 3]}


@pytest.fixture
def pd_df(dgp):
    return pd.DataFrame(dgp["df"])


@pytest.fixture
def fo_obj(dgp):
    t_col = [c for c in dgp["df"] if "T" in c][0]
    return FastOLS(
        Y=[c for c in dgp["df"].keys() if "Y" in c],
        T=t_col,
        G=[c for c in dgp["df"].keys() if "G" in c],
        X=[c for c in dgp["df"].keys() if "X" in c],
        W=[c for c in dgp["df"].keys() if "W" in c],
        discrete_treatment=True if "bin" in t_col else False,
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
            xformula="+W1**2",
            discrete_treatment=discrete_treatment,
            engine="cpu",
        )
        assert fo.Y == ["Y1", "Y2"]
        assert fo.T == "T"
        assert fo.G == ["G1", "G2"]
        assert fo.X == ["X1"]
        assert fo.W == ["W1"]
        assert fo._discrete_treatment is discrete_treatment
        assert fo._engine == "cpu"
        assert fo._fitted is False
        if discrete_treatment:
            assert (
                fo.formula.replace(" ", "")
                == "Y1+Y2~C(T)+C(G1)*C(T)+C(G2)*C(T)+X1*C(T)+W1+W1**2"
            )
        else:
            assert (
                fo.formula.replace(" ", "") == "Y1+Y2~T+C(G1)*T+C(G2)*T+X1*T+W1+W1**2"
            )

        summary = (
            "================== FastOLS Object ==================\n"
            + f"Engine: {fo._engine}\n"
            + f"Outcome Variable: {fo.Y}\n"
            + f"Treatment Variable: {fo.T}\n"
            + f"Discrete Treatment: {fo._discrete_treatment}\n"
            + f"Group Variables: {fo.G}\n"
            + f"Features/Confounders for Heterogeneity (X): {fo.X}\n"
            + f"Features/Confounders as Controls (W): {fo.W}\n"
            + f"Formula: {fo.formula}\n"
        )
        assert str(fo) == summary

        for a in ["params", "vcv", "std_err", "treatment_effects"]:
            with pytest.raises(RuntimeError):
                getattr(fo, a)

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
            assert fo._engine == "cpu"


IS_WIN_PY312 = sys.platform.startswith("win") and sys.version_info[:2] == (3, 12)


@pytest.mark.skipif(
    IS_WIN_PY312,
    reason="PySpark toPandas on Windows with Python 3.12 is unstable in CI",
)
def test__convert_dataframe_to_pandas(df_fixture):
    """Test conversion of different DataFrame types to pandas."""
    df_fxt = df_fixture
    fo_obj = FastOLS(Y=["Y"], T="T")

    if isinstance(df_fixture, dict):
        with pytest.raises(ValueError):
            fo_obj._convert_dataframe_to_pandas(df_fxt, groups=["G1", "G2"])
    else:
        df = fo_obj._convert_dataframe_to_pandas(df_fxt, groups=["G1", "G2"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (N, len(df_fxt.columns))
        assert sorted(df.columns) == sorted(df_fxt.columns)
        assert df["G1"].dtype == "category"
        assert df["G2"].dtype == "category"
        t_col = [c for c in df.columns if "T" in c][0]
        if "bin" in t_col:
            assert df[t_col].dtype == "int64"
        else:
            assert df[t_col].dtype == "float64"
        assert df["W"].dtype == "float64"
        assert df["X1"].dtype == "float64"
        assert df["X2"].dtype == "float64"
        assert df["Y1"].dtype == "float64"
        assert df["Y2"].dtype == "float64"


class TestFastOLSFittingAndEstimation:
    @pytest.fixture(autouse=True)
    def setup(self, backend):
        self.backend = backend

    @pytest.mark.parametrize("robust_vcv", [True, False], ids=["Robust", "Non-Robust"])
    @pytest.mark.parametrize(
        "estimate_effects", [True, False], ids=["Effects", "No Effects"]
    )
    def test_fit(self, fo_obj, pd_df, robust_vcv, estimate_effects):
        """Test fit method using statsmodels ols as benchmark."""
        fo_obj.fit(pd_df, estimate_effects=estimate_effects, robust_vcv=robust_vcv)
        assert fo_obj._fitted

        for k in ["params", "vcv", "std_err", "treatment_effects"]:
            getattr(fo_obj, k)

        for i, y in enumerate([c for c in pd_df.columns if "Y" in c]):
            statsmod = ols(formula=f"{y} ~ {fo_obj.formula.split('~')[1]}", data=pd_df)

            if robust_vcv:
                statsmod = statsmod.fit(cov_type="HC0")
            else:
                statsmod = statsmod.fit()

            assert np.allclose(fo_obj.params[:, i], statsmod.params)
            assert np.allclose(fo_obj.vcv[i, :, :], statsmod.cov_params())
            assert np.allclose(fo_obj.std_err[:, i], statsmod.bse)

    def test_fit_with_non_binary_discrete_treatment_raises(
        self, pd_df, fo_obj, request
    ):
        t_col = [c for c in pd_df.columns if "T" in c][0]
        if "cont" in t_col:
            pytest.skip("Not applicable for continuous treatments.")

        # Make sure non-binary discrete treatments throw an error (not supported yet)
        random_indices = np.random.choice(pd_df.index, size=10, replace=False)
        pd_df.loc[random_indices, t_col] = 3

        with pytest.raises(ValueError):
            fo_obj.fit(pd_df)

    def test_fit_with_no_groups(self, pd_df, fo_obj):
        # No passed groups will return no group treatment effects
        fo_obj.G = None
        fo_obj.__init__(
            **{
                k: getattr(fo_obj, v)
                for k, v in {
                    "Y": "Y",
                    "T": "T",
                    "G": "G",
                    "X": "X",
                    "W": "W",
                    "discrete_treatment": "_discrete_treatment",
                    "engine": "_engine",
                }.items()
            }
        )

        fo_obj.fit(pd_df, estimate_effects=True)

        for k, _ in fo_obj.treatment_effects.items():
            assert "overall" in k

    def test_fit_with_nans_raises(self, pd_df, fo_obj):
        n_nans = np.random.randint(0, len(pd_df))
        nan_indices = np.random.choice(pd_df.index, size=n_nans, replace=False)
        pd_df.loc[nan_indices, "X1"] = np.nan

        with pytest.raises(ValueError):
            fo_obj.fit(pd_df)

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

            # Check large enough R2
            r2 = r2_score(cate_estimated, cate_expected)
            assert r2 > 0.9

            # Check small enough Precision in Estimating Heterogenous Treatment Effects (PEHE)
            assert mean_squared_error(cate_estimated, cate_expected) < 0.1

    def test_predict_outcome(self, fo_obj, pd_df):
        fo_obj.fit(pd_df, estimate_effects=False)
        res = fo_obj.predict(pd_df, mode="outcome")

        for i, y in enumerate([c for c in pd_df.columns if "Y" in c]):
            y_estimated = res[:, i]
            y_expected = pd_df[y]
            assert r2_score(y_estimated, y_expected) > 0.9
            assert mean_squared_error(y_estimated, y_expected) < 0.1

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
        t_col = [c for c in pd_df.columns if "T" in c][0]
        fo_obj.fit(pd_df, estimate_effects=True)

        if custom_GATE:
            res = fo_obj.estimate_ate(
                pd_df,
                return_results_dict=True,
            )
            prettified = fo_obj.prettify_treatment_effects(res)
        else:
            prettified = fo_obj.prettify_treatment_effects()
            res = fo_obj.treatment_effects

        assert isinstance(prettified, pd.DataFrame)

        assert "group" in prettified.columns
        assert "membership" in prettified.columns
        assert "outcome" in prettified.columns
        if "bin" in t_col:
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
            for _, v in res.items():
                stack = (
                    np.hstack([stack, v[c].flatten()])
                    if stack is not None
                    else v[c].flatten()
                )

            assert np.allclose(stack, prettified[c])
