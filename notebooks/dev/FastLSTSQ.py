import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# FastLSTSQ API Usage""")
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from caml import FastLSTSQ
    return FastLSTSQ, np, pd


@app.cell
def _(np, pd):
    np.random.seed(10)

    df = pd.DataFrame({'Y1': np.random.normal(size=2_500_000),
                       'Y2': np.random.normal(size=2_500_000),
                       'Y3': np.random.normal(size=2_500_000),
                       'T':  np.random.randint(0,2,size=2_500_000),
                       'G1': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'G2': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'G3': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'G4': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'G5': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'G6': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'G7': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'G8': np.random.randint(0,5,size=2_500_000).astype('str'),
                       'X1': np.random.normal(size=2_500_000),
                       'X2': np.random.normal(size=2_500_000),
                       'X3': np.random.normal(size=2_500_000),
                       'X4': np.random.normal(size=2_500_000),
                       'X5': np.random.normal(size=2_500_000),
                       'X6': np.random.normal(size=2_500_000),
                       'X7': np.random.normal(size=2_500_000),
                       'X8': np.random.normal(size=2_500_000),
                       'X9': np.random.normal(size=2_500_000),
                       'product_group': np.random.choice(['total','produce'],size=2_500_000),
                       'modality': np.random.choice(['all','online'],size=2_500_000)})
    return (df,)


@app.cell
def _(FastLSTSQ, df):
    fu = FastLSTSQ(Y=[c for c in df.columns if "Y" in c],
                    T="T",
                    G=[c for c in df.columns if "G" in c],
                    X=[c for c in df.columns if "X" in c],
                    engine='cpu')
    return (fu,)


@app.cell
def _(df, fu):
    fu.fit_and_estimate(data=df, parallel=True)
    return


@app.cell
def _(FastLSTSQ, df):
    import timeit

    cpu_times = []
    gpu_times = []

    fu1 = FastLSTSQ(Y=[c for c in df.columns if "Y" in c],
                    T="T",
                    G=[c for c in df.columns if "G" in c],
                    X=[c for c in df.columns if "X" in c],
                    engine='cpu')
    fu1.fit_and_estimate(data=df)

    del fu1
    for _ in range(5):
        start_time = timeit.default_timer()
        fu2 = FastLSTSQ(Y=[c for c in df.columns if "Y" in c],
                    T="T",
                    G=[c for c in df.columns if "G" in c],
                    X=[c for c in df.columns if "X" in c],
                    engine='cpu')
        fu2.fit_and_estimate(data=df)
        cpu_times.append(timeit.default_timer() - start_time)

    del fu2

    fu3 = FastLSTSQ(Y=[c for c in df.columns if "Y" in c],
                    T="T",
                    G=[c for c in df.columns if "G" in c],
                    X=[c for c in df.columns if "X" in c],
                    engine='gpu')
    fu3.fit_and_estimate(data=df)

    del fu3

    for _ in range(5):
        start_time = timeit.default_timer()
        fu4 = FastLSTSQ(Y=[c for c in df.columns if "Y" in c],
                    T="T",
                    G=[c for c in df.columns if "G" in c],
                    X=[c for c in df.columns if "X" in c],
                    engine='gpu')
        fu4.fit_and_estimate(data=df)
        gpu_times.append(timeit.default_timer() - start_time)

    del fu4

    print(f"CPU times: {cpu_times}")
    print(f"GPU times: {gpu_times}")
    print(f"Average CPU time: {sum(cpu_times)/len(cpu_times):.4f} seconds")
    print(f"Average GPU time: {sum(gpu_times)/len(gpu_times):.4f} seconds")
    return cpu_times, fu1, fu2, fu3, fu4, gpu_times, start_time, timeit


if __name__ == "__main__":
    app.run()
