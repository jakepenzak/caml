import marimo

__generated_with = "0.10.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Caml Synthetic Data API Usage""")
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import CamlSyntheticDataGenerator
    return (CamlSyntheticDataGenerator,)


@app.cell
def _(CamlSyntheticDataGenerator):
    data =  CamlSyntheticDataGenerator(n_obs=10_000,
                                      n_cont_outcomes=1,
                                      n_binary_outcomes=1,
                                      n_cont_treatments=1,
                                      n_binary_treatments=1,
                                      n_discrete_treatments=0,
                                      n_cont_confounders=2,
                                      n_binary_confounders=2,
                                      n_discrete_confounders=2,
                                      n_cont_heterogeneity_covariates=4,
                                      n_binary_heterogeneity_covariates=4,
                                      n_discrete_heterogeneity_covariates=4,
                                      n_heterogeneity_confounders=0,
                                      stddev_outcome_noise=3,
                                      stddev_treatment_noise=3,
                                      causal_model_functional_form="fully_non_linear",
                                      n_nonlinear_transformations=10,
                                      n_nonlinear_interactions=5,
                                      treatment_effect_weight=1,
                                      seed=None)
    return (data,)


@app.cell
def _(data):
    data.df
    return


@app.cell
def _(data):
    data.cates
    return


@app.cell
def _(data):
    data.ates
    return


if __name__ == "__main__":
    app.run()
