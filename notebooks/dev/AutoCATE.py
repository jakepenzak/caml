import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# AutoCate API Usage""")
    return


@app.cell
def _():
    from caml.logging import configure_logging
    import logging

    configure_logging(level=logging.DEBUG)
    return


@app.cell
def _(mo):
    mo.md("""## Synthetic Data""")
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data = SyntheticDataGenerator(
        n_obs=1_000,
        n_cont_outcomes=1,
        n_binary_outcomes=0,
        n_binary_treatments=1,
        n_discrete_treatments=0,
        n_cont_treatments=0,
        n_cont_confounders=3,
        n_binary_confounders=3,
        n_discrete_confounders=0,
        n_cont_modifiers=3,
        n_binary_modifiers=3,
        n_discrete_modifiers=2,
        stddev_outcome_noise=1,
        stddev_treatment_noise=1,
        causal_model_functional_form="linear",
        seed=None,
    )


    synthetic_df = data.df
    cate_df = data.cates
    ate_df = data.ates
    dgp = data.dgp
    return ate_df, cate_df, dgp, synthetic_df


@app.cell
def _(synthetic_df):
    synthetic_df
    return


@app.cell
def _(dgp):
    dgp
    return


@app.cell
def _(cate_df):
    cate_df
    return


@app.cell
def _(ate_df):
    ate_df
    return


@app.cell
def _(synthetic_df):
    for col in synthetic_df.columns:
        if "discrete" in col:
            synthetic_df[col] = synthetic_df[col].astype("category")

    categories = list(synthetic_df["X7_discrete"].cat.categories)
    mapping = dict(
        zip(
            categories,
            [f"A{str(i)}" for i in categories],
        )
    )
    synthetic_df["X7_discrete"] = synthetic_df["X7_discrete"].map(mapping)
    return


@app.cell
def _(synthetic_df):
    synthetic_df
    return


@app.cell
def _(mo):
    mo.md("""## Core API""")
    return


@app.cell
def _(mo):
    mo.md("""### AutoCATE""")
    return


@app.cell
def _(mo):
    mo.md("""#### Class Instantiation""")
    return


@app.cell
def _(synthetic_df):
    from caml import AutoCATE

    outcome = [c for c in synthetic_df.columns if "Y" in c][0]
    treatment = [c for c in synthetic_df.columns if "T" in c][0]

    caml = AutoCATE(
        Y=outcome,
        T=treatment,
        X=[c for c in synthetic_df.columns if "X" in c]
        + [c for c in synthetic_df.columns if "W" in c],
        W=[],
        discrete_treatment=True
        if "binary" in treatment or "discrete" in treatment
        else False,
        discrete_outcome=True if "binary" in outcome else False,
        model_Y={"time_budget": 5},
        model_T={"time_budget": 5},
        model_regression={"time_budget": 5},
        enable_categorical=True,
        n_jobs=-1,
        use_ray=False,
        ray_remote_func_options_kwargs=None,
        use_spark=False,
        seed=None,
    )
    return (caml,)


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")
    return


@app.cell
def _(caml, synthetic_df):
    caml.fit(synthetic_df)
    return


@app.cell
def _(caml, synthetic_df):
    cate_predictions = caml.estimate_cate(synthetic_df)
    caml.estimate_ate(synthetic_df)
    return (cate_predictions,)


@app.cell
def _(caml, synthetic_df):
    obj2 = caml.estimate_cate(synthetic_df, return_inference=True)
    return (obj2,)


@app.cell
def _(caml, synthetic_df):
    caml.estimate_ate(synthetic_df, return_inference=True)
    return


@app.cell
def _(obj2):
    obj2
    return


@app.cell
def _(ate_df):
    ate_df
    return


@app.cell
def _(mo):
    mo.md("""# Plots""")
    return


@app.cell
def _(cate_df):
    from caml.extensions.plots import (
        cate_histogram_plot,
        cate_true_vs_estimated_plot,
        cate_line_plot,
    )

    true_cates = cate_df.to_numpy()
    return (
        cate_histogram_plot,
        cate_line_plot,
        cate_true_vs_estimated_plot,
        true_cates,
    )


@app.cell
def _(cate_predictions, cate_true_vs_estimated_plot, true_cates):
    cate_true_vs_estimated_plot(
        true_cates=true_cates, estimated_cates=cate_predictions
    )
    return


@app.cell
def _(cate_predictions):
    cate_predictions.shape
    return


@app.cell
def _(cate_histogram_plot, cate_predictions):
    cate_histogram_plot(estimated_cates=cate_predictions)
    return


@app.cell
def _(cate_histogram_plot, cate_predictions, true_cates):
    cate_histogram_plot(estimated_cates=cate_predictions, true_cates=true_cates)
    return


@app.cell
def _(cate_line_plot, cate_predictions):
    cate_line_plot(estimated_cates=cate_predictions.flatten(), window=10)
    return


@app.cell
def _(cate_line_plot, cate_predictions, true_cates):
    cate_line_plot(
        estimated_cates=cate_predictions.flatten(),
        true_cates=true_cates.flatten(),
        window=10,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
