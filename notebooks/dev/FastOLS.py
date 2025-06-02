import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# FastOLS API Usage""")
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from caml import FastOLS
    from caml.extensions.synthetic_data import SyntheticDataGenerator
    from caml.logging import configure_logging
    import logging

    configure_logging(level=logging.DEBUG)
    return FastOLS, SyntheticDataGenerator


@app.cell
def _(SyntheticDataGenerator):
    data_generator = SyntheticDataGenerator(
        n_obs=10_000,
        n_cont_outcomes=2,
        n_binary_outcomes=0,
        n_cont_treatments=0,
        n_binary_treatments=1,
        n_discrete_treatments=0,
        n_cont_confounders=1,
        n_cont_modifiers=1,
        n_binary_modifiers=2,
        n_discrete_modifiers=1,
        stddev_outcome_noise=1,
        stddev_treatment_noise=1,
        causal_model_functional_form="linear",
        seed=None,
    )
    return (data_generator,)


@app.cell
def _(data_generator):
    df = data_generator.df
    # df["cates"] = data_generator.cates
    df
    return (df,)


@app.cell
def _(data_generator):
    data_generator.dgp
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""## Fit w/ Effect Estimation in One Pass""")
    return


@app.cell
def _(FastOLS, df):
    fu = FastOLS(
        Y=[c for c in df.columns if "Y" in c],
        T="T1_binary",
        G=[c for c in df.columns if "X" in c and ("bin" in c or "dis" in c)],
        X=[c for c in df.columns if "X" in c and "cont" in c],
        W=[c for c in df.columns if "W" in c],
        xformula="+W1_continuous**2",
        engine="gpu",
        discrete_treatment=True,
    )
    return (fu,)


@app.cell
def _(fu):
    fu.formula
    return


@app.cell
def _(df, fu):
    fu.fit(df, n_jobs=-1, estimate_effects=True, robust_vcv=False)
    return


@app.cell
def _(fu):
    fu.params
    return


@app.cell
def _(fu):
    fu.vcv
    return


@app.cell
def _(fu):
    fu.std_err
    return


@app.cell
def _(fu):
    fu.treatment_effects
    return


@app.cell
def _(df, fu):
    cates = fu.estimate_cate(df, return_results_dict=False)

    cates
    return


@app.cell
def _(df, fu):
    cate_predictions = fu.predict(df)
    cate_predictions
    return


@app.cell
def _(data_generator):
    data_generator.cates
    return


@app.cell
def _(df, fu):
    fu.predict(df, mode="y")
    return


@app.cell
def _(fu):
    fu.prettify_treatment_effects()
    return


@app.cell
def _(data_generator):
    data_generator.ates
    return


@app.cell
def _(df, fu):
    for g in fu.G:
        M = df[g].unique().tolist()
        for m in M:
            print(f"ATE for {g}, {m}: {df[df[f'{g}'] == m]['cates'].mean()}")
    return


@app.cell
def _(df, fu):
    df2 = df.query(
        "X2_binary == 0 & X4_discrete == 3 & X1_continuous < 5"
    ).copy()

    fu.prettify_treatment_effects(effects=fu.estimate_ate(df2, return_results_dict=True))
    return (df2,)


@app.cell
def _(df2):
    df2['cates'].mean()
    return


if __name__ == "__main__":
    app.run()
