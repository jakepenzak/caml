import pytest

import caml.core.ols as ols_mod
from caml import FastOLS

pytestmark = [pytest.mark.core, pytest.mark.ols]


class TestFastOLSInitialization:
    @pytest.mark.parametrize("discrete_treatment", [True, False])
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
