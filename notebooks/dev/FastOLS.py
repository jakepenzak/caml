

import marimo

__generated_with = "0.13.1"
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
    from caml.extensions.synthetic_data import CamlSyntheticDataGenerator
    from caml.logging import configure_logging
    import logging

    configure_logging(level=logging.DEBUG)
    return CamlSyntheticDataGenerator, FastOLS


@app.cell
def _(CamlSyntheticDataGenerator):
    data = CamlSyntheticDataGenerator(
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
        stddev_outcome_noise=10,
        stddev_treatment_noise=10,
        causal_model_functional_form="linear",
        seed=44,
    )
    return (data,)


@app.cell
def _(data):
    df = data.df
    # df["cates"] = data.cates
    df
    return (df,)


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
        engine="gpu",
        discrete_treatment=True,
    )
    return (fu,)


@app.cell
def _(df, fu):
    fu.fit(df, n_jobs=-1, estimate_effects=True, robust_vcv=False)

    fu.results
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
def _(data):
    data.cates
    return


@app.cell
def _(fu):
    d = fu.prettify_treatment_effects()
    return (d,)


@app.cell
def _(d):
    d
    return


@app.cell
def _(d):
    d.query("group == 'overall'")['ate']
    return


@app.cell
def _(data):
    data.ates
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
