import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    from caml.data import CausalDataset, OutcomeType, TreatmentType
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(n_cont_modifiers=3, n_cont_confounders=4)

    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        W=[c for c in gen.df.columns if "W" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
        true_cates=np.array(gen.cates),
    )
    return data, np


@app.cell
def _(data):
    from caml.nuisance import NuisanceTuner, NuisanceTunerSpec

    spec = NuisanceTunerSpec(
        fit_treatment_model=True, fit_outcome_model=True, fit_regression_model=True
    )

    tuner = NuisanceTuner(time_budget=10, verbose=0)

    tuner.fit(data, spec)

    print(f"\n{tuner.treatment_model_}")
    print(f"\n{tuner.outcome_model_}")
    print(f"\n{tuner.regression_model_}")
    return (tuner,)


@app.cell
def _(data, tuner):
    from caml.samplers.cross_fit import CrossFitter

    cross_fitter = CrossFitter(cv=5)

    mhat, ehat = cross_fitter.fit_predict_nuisances_dml(
        data, outcome_model=tuner.outcome_model_, treatment_model=tuner.treatment_model_
    )
    return (cross_fitter,)


@app.cell
def _(data, tuner):
    from econml.dml import LinearDML

    mod = LinearDML(
        model_y=tuner.outcome_model_,
        model_t=tuner.treatment_model_,
        discrete_treatment=True,
    )

    mod.fit(data.Y, data.T, X=data.X, W=data.W)
    return (mod,)


@app.cell
def _(data, mod):
    from caml.scorers.pehe import PEHE

    pehe = PEHE(normalized=True)

    pehe(estimator=mod, data=data)
    return


@app.cell
def _(data, mod, tuner):
    from caml.scorers.r_loss import RLoss

    r_loss = RLoss(
        treatment_model=tuner.treatment_model_,
        outcome_model=tuner.outcome_model_,
        normalized=True,
    )

    r_loss(estimator=mod, data=data)
    return


@app.cell
def _(data, mod, tuner):
    from caml.scorers.q_loss import QLoss

    q_loss = QLoss(treatment_model=tuner.treatment_model_)

    q_loss(estimator=mod, data=data)
    return


@app.cell
def _(data, np):
    XW = np.hstack([data.X, data.W]) if data.W is not None else data.X
    return (XW,)


@app.cell
def _(XW):
    XW.shape
    return


@app.cell
def _(XW, data):
    XW[data.T == 1]
    return


@app.cell
def _(data, mod, tuner):
    from caml.scorers.dr_loss import DRLoss

    dr_loss = DRLoss(
        treatment_model=tuner.treatment_model_,
        regression_model=tuner.regression_model_,
        normalized=True,
    )

    dr_loss(estimator=mod, data=data)
    return


@app.cell
def _(data, mod):
    from caml.extensions.plots import (
        cate_histogram_plot,
        cate_line_plot,
        cate_true_vs_estimated_plot,
    )

    estimated = mod.effect(data.X)

    cate_histogram_plot(estimated, true_cates=data.true_cates)
    return cate_line_plot, cate_true_vs_estimated_plot, estimated


@app.cell
def _(cate_line_plot, data, estimated):
    cate_line_plot(estimated.ravel(), true_cates=data.true_cates.ravel())
    return


@app.cell
def _(cate_true_vs_estimated_plot, data, estimated):
    cate_true_vs_estimated_plot(estimated.ravel(), data.true_cates.ravel())
    return


@app.cell
def _(cross_fitter, data, mod, tuner):
    e_hat = cross_fitter.fit_predict_treatment_model(
        data=data,
        treatment_model=tuner.treatment_model_,
    )

    ipw = (data.T * data.Y) / e_hat - ((1 - data.T) * data.Y) / (1 - e_hat)
    tau_hat = mod.effect(data.X)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
