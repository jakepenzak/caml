import marimo

__generated_with = "0.10.18"
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
    from caml.logging import configure_logging
    import logging

    configure_logging(level=logging.DEBUG)
    return FastLeastSquares, configure_logging, logging, np, pd


@app.cell
def _(np, pd):
    np.random.seed(11)

    df = pd.DataFrame({'Y1': np.random.normal(size=10_000),
                       'Y2': np.random.normal(size=10_000),
                       'Y3': np.random.normal(size=10_000),
                       'T':  np.random.randint(0,2,size=10_000),
                       'G1': np.random.randint(0,5,size=10_000).astype('str'),
                       'G2': np.random.randint(0,5,size=10_000).astype('str'),
                       'G3': np.random.randint(0,5,size=10_000).astype('str'),
                       'G4': np.random.randint(0,5,size=10_000).astype('str'),
                       'G5': np.random.randint(0,5,size=10_000).astype('str'),
                       'G6': np.random.randint(0,5,size=10_000).astype('str'),
                       'G7': np.random.randint(0,5,size=10_000).astype('str'),
                       'G8': np.random.randint(0,5,size=10_000).astype('str'),
                       'X1': np.random.normal(size=10_000),
                       'X2': np.random.normal(size=10_000),
                       'X3': np.random.normal(size=10_000),
                       'W1': np.random.normal(size=10_000),
                       'W2': np.random.normal(size=10_000),
                       'W3': np.random.normal(size=10_000),
                       'W4': np.random.normal(size=10_000),
                       'W5': np.random.normal(size=10_000),
                       'W6': np.random.normal(size=10_000),
                       'product_group': np.random.choice(['total','produce'],size=10_000),
                       'modality': np.random.choice(['all','online'],size=10_000)})
    return (df,)


@app.cell
def _(FastLeastSquares, df):
    fu = FastLeastSquares(Y=[c for c in df.columns if "Y" in c],
                    T="T",
                    G=[c for c in df.columns if "G" in c],
                    X=[c for c in df.columns if "X" in c],
                    W=[c for c in df.columns if "W" in c],
                    engine='cpu')
    return (fu,)


@app.cell
def _(df, fu):
    fu.fit(data=df, n_jobs=-1)
    return


@app.cell
def _(df, fu):
    df2 = df.query("G1 == '1' & G2 == '2' & X1 > 0").copy()

    fu.estimate_single_cate(data=df2)
    return (df2,)


@app.cell
def _(fu):
    fu.prettify_treatment_effects()
    return


if __name__ == "__main__":
    app.run()
