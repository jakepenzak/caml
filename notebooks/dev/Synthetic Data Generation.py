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


    def gen():
        data_generator = SyntheticDataGenerator(
            n_obs=10_000,
            n_cont_outcomes=1,
            n_binary_outcomes=1,
            n_cont_treatments=0,
            n_binary_treatments=1,
            n_discrete_treatments=0,
            n_cont_confounders=2,
            n_binary_confounders=2,
            n_discrete_confounders=0,
            n_cont_modifiers=1,
            n_binary_modifiers=1,
            n_discrete_modifiers=0,
            n_confounding_modifiers=0,
            stddev_outcome_noise=1,
            stddev_treatment_noise=1,
            causal_model_functional_form="linear",
            n_nonlinear_transformations=10,
            seed=None,
        )

        df = data_generator.df
        dgp = data_generator.dgp
        ates = data_generator.ates
        cates = data_generator.cates

        return data_generator, df, dgp, ates, cates
    return (gen,)


@app.cell
def _(gen):
    data_generator, df, dgp, ates, cates = gen()
    return ates, cates, data_generator, df, dgp


@app.cell
def _():
    # import numpy as np

    # ates_list = []
    # for i in range(20):
    #     _, _, _, ate, _ = gen()
    #     for a in ate.iloc[:, 1]:
    #         ates_list.append(abs(a))

    # np.mean(ates_list)
    return


@app.cell
def _(ates):
    ates
    return


@app.cell
def _(cates):
    cates
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(dgp):
    dgp
    return


@app.cell
def _(data_generator, df, dgp):
    y1_dgp = dgp["Y2_binary"]

    design_matrix = data_generator.create_design_matrix(
        df, formula=y1_dgp["formula"], return_type="dataframe"
    )


    # Recreate Y1_continuous
    params = y1_dgp["params"]
    noise = y1_dgp["noise"]
    f = y1_dgp["function"]

    import pandas as pd

    dff = pd.DataFrame(f(design_matrix, params, noise))

    from matplotlib import pyplot as plt

    dff.hist()
    plt.show()
    return


@app.cell
def _():
    # r
    return


if __name__ == "__main__":
    app.run()
