import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# FastLeastSquares API Usage""")
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from caml import FastLeastSquares
    from caml.extensions.synthetic_data import CamlSyntheticDataGenerator
    from caml.logging import configure_logging
    import logging

    configure_logging(level=logging.DEBUG)
    return (
        CamlSyntheticDataGenerator,
        FastLeastSquares,
        configure_logging,
        logging,
        np,
        pd,
    )


@app.cell
def _(CamlSyntheticDataGenerator):
    data = CamlSyntheticDataGenerator(n_obs=10_000,
                                      n_cont_outcomes=1,
                                      n_binary_outcomes=0,
                                      n_cont_treatments=0,
                                      n_binary_treatments=1,
                                      n_cont_confounders=2,
                                      n_cont_modifiers=2,
                                      n_discrete_modifiers=2,
                                      stddev_outcome_noise=1,
                                      stddev_treatment_noise=1,
                                      causal_model_functional_form="linear",
                                      seed=0)
    return (data,)


@app.cell
def _(data):
    df = data.df
    df['cates'] = data.cates
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(FastLeastSquares, df):
    fu = FastLeastSquares(Y=[c for c in df.columns if "Y" in c],
                    T="T1_binary",
                    G=[c for c in df.columns if "X" in c and "disc" in c],
                    X=[c for c in df.columns if "X" in c and "cont" in c],
                    W=[c for c in df.columns if "W" in c],
                    engine='cpu',
                    discrete_treatment=True)
    return (fu,)


@app.cell
def _(df, fu):
    fu.fit(data=df, n_jobs=-1, estimate_gates=True)
    return


@app.cell
def _(fu):
    fu.prettify_treatment_effects()
    return


@app.cell
def _(df, fu):
    for g in fu.G:
        M = df[g].unique().tolist()
        for m in M:
            print(f"ATE for {g}, {m}: {df[df[f'{g}']==m]['cates'].mean()}")
    return M, g, m


@app.cell
def _(df, fu):
    df2 = df.query("X3_discrete == 4 & X4_discrete == 2 & X1_continuous < 0").copy()

    fu.estimate_single_cate(data=df2)
    return (df2,)


if __name__ == "__main__":
    app.run()
