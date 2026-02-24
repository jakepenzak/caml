import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from caml.data import CausalDataset, OutcomeType, TreatmentType
    from caml.utilities.synthetic_data import SyntheticDataGenerator
    from caml import configure_logging

    configure_logging(verbose=2)

    gen = SyntheticDataGenerator(n_obs=1_000,
                                 n_cont_modifiers=3,
                                 n_binary_modifiers=2,
                                 n_cont_confounders=2,
                                 n_binary_confounders=2,
                                 n_binary_outcomes=0,
                                 n_cont_outcomes=1,
                                 n_binary_treatments=1,
                                 n_cont_treatments=0,
                                 n_confounding_modifiers=2,
                                 causal_model_functional_form="nonlinear",
                                 seed=10)

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
    return (data,)


@app.cell
def _(data):
    from caml.registry import get_compatible_estimators

    candidates = list(get_compatible_estimators(data=data, families=["dml","dr","meta"]))

    candidates
    return (get_compatible_estimators,)


@app.cell
def _(data, get_compatible_estimators):
    from caml import AutoCATE
    from caml.automl import OptunaBackend


    mod = AutoCATE(nuisance_time_budget_s=5,
                   n_jobs=1,
                   n_trials=5,
                  verbose=2,
                  candidate_cate_estimators=list(get_compatible_estimators(data=data, families=["dml","dr","meta"])))

    mod.fit(data)
    return (mod,)


@app.cell
def _(data, mod):
    mod.best_estimator_.fit(data.sample(mod.train_indices))
    return


@app.cell
def _(data, mod):
    from caml.utilities.plots import (
        cate_histogram_plot,
        cate_line_plot,
        cate_true_vs_estimated_plot,
    )

    estimated = mod.best_estimator_.effect(data.sample(mod.test_indices).X)
    true_cates = data.sample(mod.test_indices).true_cates

    cate_histogram_plot(estimated, true_cates=true_cates)
    return cate_line_plot, cate_true_vs_estimated_plot, estimated, true_cates


@app.cell
def _(cate_line_plot, estimated, true_cates):
    cate_line_plot(estimated.ravel(), true_cates=true_cates.ravel())
    return


@app.cell
def _(cate_true_vs_estimated_plot, estimated, true_cates):
    cate_true_vs_estimated_plot(estimated.ravel(), true_cates.ravel())
    return


@app.cell
def _(data, mod):
    from caml.scorers import Pehe

    pehe = Pehe(normalized=True)

    pehe(estimator=mod.best_estimator_, data=data.sample(mod.test_indices))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
