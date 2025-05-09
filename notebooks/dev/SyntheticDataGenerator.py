import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Caml Synthetic Data API Usage""")
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import SyntheticDataGenerator


    data_generator = SyntheticDataGenerator(
        n_obs=10_000,
        n_cont_outcomes=1,
        n_binary_outcomes=1,
        n_cont_treatments=0,
        n_binary_treatments=1,
        n_discrete_treatments=1,
        n_cont_confounders=2,
        n_binary_confounders=2,
        n_discrete_confounders=0,
        n_cont_modifiers=2,
        n_binary_modifiers=2,
        n_discrete_modifiers=0,
        n_confounding_modifiers=0,
        stddev_outcome_noise=1,
        stddev_treatment_noise=1,
        causal_model_functional_form="nonlinear",
        n_nonlinear_transformations=10,
        seed=None,
    )

    df = data_generator.df
    dgp = data_generator.dgp
    ates = data_generator.ates
    cates = data_generator.cates
    return ates, cates, df, dgp


@app.cell
def _(df):
    df
    return


@app.cell
def _(dgp):
    dgp
    return


@app.cell
def _(ates):
    ates
    return


@app.cell
def _(cates):
    cates
    return


if __name__ == "__main__":
    app.run()
