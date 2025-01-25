import marimo

__generated_with = "0.10.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Caml Synthetic Data API Usage""")
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import CamlSyntheticDataGenerator
    return (CamlSyntheticDataGenerator,)


@app.cell
def _(mo):
    mo.md(r"""## Generate Data""")
    return


@app.cell
def _(CamlSyntheticDataGenerator):
    data =  CamlSyntheticDataGenerator(n_obs=10_000,
                                      n_cont_outcomes=1,
                                      n_binary_outcomes=1,
                                      n_cont_treatments=1,
                                      n_binary_treatments=1,
                                      n_discrete_treatments=1,
                                      n_cont_confounders=1,
                                      n_binary_confounders=1,
                                      n_discrete_confounders=1,
                                      n_cont_modifiers=1,
                                      n_binary_modifiers=1,
                                      n_discrete_modifiers=1,
                                      n_confounding_modifiers=1,
                                      stddev_outcome_noise=3,
                                      stddev_treatment_noise=3,
                                      causal_model_functional_form="linear",
                                      n_nonlinear_transformations=10,
                                      n_nonlinear_interactions=5,
                                      seed=15)
    return (data,)


@app.cell
def _(mo):
    mo.md(r"""## Simulated Dataframe""")
    return


@app.cell
def _(data):
    data.df
    return


@app.cell
def _(mo):
    mo.md(r"""## DGP""")
    return


@app.cell
def _(data):
    for k,v in data.dgp.items():
        print(k)
        print(v)
    return k, v


@app.cell
def _(mo):
    mo.md(r"""## True Conditional Average Treatment Effects (CATEs)""")
    return


@app.cell
def _(data):
    data.cates
    return


@app.cell
def _(mo):
    mo.md(r"""## True Average Treatment Effects (ATEs)""")
    return


@app.cell
def _(data):
    data.ates
    return


if __name__ == "__main__":
    app.run()
