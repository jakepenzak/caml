import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Caml Synthetic Data Generator""")
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    import numpy as np
    return SyntheticDataGenerator, np


@app.cell
def _(mo):
    mo.md(r"""## Generate Data""")
    return


@app.cell
def _(SyntheticDataGenerator):
    data_generator = SyntheticDataGenerator(
        n_obs=10_000,
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
        n_nonlinear_transformations=5,
        seed=10,
    )
    return (data_generator,)


@app.cell
def _(mo):
    mo.md(r"""## Simulated Dataframe""")
    return


@app.cell
def _(data_generator):
    data_generator.df
    return


@app.cell
def _(mo):
    mo.md(r"""## True Conditional Average Treatment Effects (CATEs)""")
    return


@app.cell
def _(data_generator):
    data_generator.cates
    return


@app.cell
def _(mo):
    mo.md(r"""## True Average Treatment Effects (ATEs)""")
    return


@app.cell
def _(data_generator):
    data_generator.ates
    return


@app.cell
def _(mo):
    mo.md(r"""## True Data Generating Process""")
    return


@app.cell
def _(data_generator):
    for k, v in data_generator.dgp.items():
        print(f"DGP for {k}:")
        print(v)
    return


@app.cell
def _(mo):
    mo.md(r"""We can recreate the raw scores of our treatment and outcome variables too:""")
    return


@app.cell
def _(data_generator, np):
    # Recreate Y1_continuous
    df = data_generator.df
    dgp = data_generator.dgp["Y1_continuous"]

    design_matrix = data_generator.create_design_matrix(df, formula=dgp["formula"])

    params = dgp["params"]
    noise = dgp["noise"]
    f = dgp["function"]

    raw_scores = f(design_matrix, params, noise)

    assert np.allclose(raw_scores, df["Y1_continuous"])
    assert np.allclose(raw_scores, dgp["raw_scores"])

    raw_scores
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""For treatment variables, we get back the probabilities:""")
    return


@app.cell
def _(data_generator, df, np):
    # Recreate Y2_binary
    dgp_bin = data_generator.dgp["Y2_binary"]

    design_matrix_bin = data_generator.create_design_matrix(df, formula=dgp_bin["formula"])

    params_bin = dgp_bin["params"]
    noise_bin = dgp_bin["noise"]
    f_bin = dgp_bin["function"]

    raw_scores_bin = f_bin(design_matrix_bin, params_bin, noise_bin)

    assert np.allclose(raw_scores_bin, dgp_bin["raw_scores"])

    raw_scores_bin
    return


if __name__ == "__main__":
    app.run()
