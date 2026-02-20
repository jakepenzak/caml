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
    return (
        CausalDataset,
        OutcomeType,
        SyntheticDataGenerator,
        TreatmentType,
        data,
        np,
    )


@app.cell
def _(data):
    from caml.nuisance import NuisanceTuner, NuisanceTunerSpec

    spec = NuisanceTunerSpec(
        fit_treatment_model=True, fit_outcome_model=True, fit_regression_model=True
    )

    tuner = NuisanceTuner(time_budget=5, verbose=0)

    tuner.fit(data, spec)

    print(f"\n{tuner.treatment_model_}")
    print(f"\n{tuner.outcome_model_}")
    print(f"\n{tuner.regression_model_}")
    return (tuner,)


@app.cell
def _(RLoss, data, tuner):
    r_losss = RLoss(
        treatment_model=tuner.treatment_model_,
        outcome_model=tuner.outcome_model_,
        normalized=False,
    )


    from caml.automl.backends.optuna import OptunaBackend
    from caml.registry import get_compatible_estimators
    optuna_backend = OptunaBackend()

    candidate_estimators = get_compatible_estimators(data, families=["dml"])

    objective = optuna_backend.create_objective(r_losss, candidate_estimators, data, tuner.outcome_model_, tuner.treatment_model_, tuner.regression_model_)

    study = optuna_backend.optimize(objective, n_trials=100, n_jobs=-1)
    return (study,)


@app.cell
def _(study):
    dir(study)
    return


@app.cell
def _(data, tuner):
    from caml.samplers.cross_fit import CrossFitter

    cross_fitter = CrossFitter(cv=5)

    mhat, ehat = cross_fitter.fit_predict_nuisances_dml(
        data, outcome_model=tuner.outcome_model_, treatment_model=tuner.treatment_model_
    )
    return


@app.cell
def _(
    CausalDataset,
    OutcomeType,
    Pehe,
    SyntheticDataGenerator,
    TreatmentType,
    np,
):
    from caml.estimators.dml import WrappedLinearDML
    from sklearn.linear_model import LinearRegression, LogisticRegression

    gener = SyntheticDataGenerator(n_cont_modifiers=3, n_cont_confounders=3, seed=10)
    df = gener.df
    true_cates = np.array(gener.cates)

    dataa = CausalDataset.from_dataframe(
        df=df,
        X=["X1_continuous", "X2_continuous", "X3_continuous"],
        W=["W1_continuous", "W2_continuous", "W3_continuous"],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
        true_cates=true_cates,
    )

    estimator = WrappedLinearDML(
        model_y=LinearRegression(), model_t=LogisticRegression(), cv=3
    )
    estimator.fit(dataa)

    scorer = Pehe()
    print(f"PEHE: {scorer(estimator, dataa)}")

    nrm_scorer = Pehe(normalized=True)
    print(f"Normalized PEHE: {nrm_scorer(estimator, dataa):.2f}")
    return dataa, estimator, true_cates


@app.cell
def _(true_cates):
    true_cates.shape
    return


@app.cell
def _(dataa, estimator):
    x = estimator.effect(dataa.X)
    y = dataa.true_cates
    return x, y


@app.cell
def _(np, x, y):
    np.mean((y - x) ** 2)
    return


@app.cell
def _(np, x):
    np.mean(x)
    return


@app.cell
def _(y):
    y
    return


@app.cell
def _(cate_histogram_plot, dataa, estimator):
    cate_histogram_plot(estimator.effect(dataa.X).ravel(), true_cates=dataa.true_cates)
    return


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
    from caml.scorers.pehe import Pehe

    pehe = Pehe(normalized=True)

    pehe(estimator=mod, data=data)
    return (Pehe,)


@app.cell
def _(data, mod, tuner):
    from caml.scorers.r_loss import RLoss

    r_loss = RLoss(
        treatment_model=tuner.treatment_model_,
        outcome_model=tuner.outcome_model_,
        normalized=True,
    )

    r_loss(estimator=mod, data=data)
    return (RLoss,)


@app.cell
def _(data, mod, tuner):
    from caml.scorers.q_stat import QStat

    q_stat = QStat(treatment_model=tuner.treatment_model_)

    q_stat(estimator=mod, data=data)
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
    return (
        cate_histogram_plot,
        cate_line_plot,
        cate_true_vs_estimated_plot,
        estimated,
    )


@app.cell
def _(cate_line_plot, data, estimated):
    cate_line_plot(estimated.ravel(), true_cates=data.true_cates.ravel())
    return


@app.cell
def _(cate_true_vs_estimated_plot, data, estimated):
    cate_true_vs_estimated_plot(estimated.ravel(), data.true_cates.ravel())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
