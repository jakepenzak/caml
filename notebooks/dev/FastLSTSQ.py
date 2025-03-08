import marimo

__generated_with = "0.10.18"
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
    df = pd.DataFrame({'Y1': np.random.normal(size=1_000_000),
                       'Y2': np.random.normal(size=1_000_000),
                       'Y3': np.random.normal(size=1_000_000),
                       'T':  np.random.randint(0,2,size=1_000_000),
                       'G1': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'G2': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'G3': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'G4': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'G5': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'G6': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'G7': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'G8': np.random.randint(0,5,size=1_000_000).astype('str'),
                       'X1': np.random.normal(size=1_000_000),
                       'X2': np.random.normal(size=1_000_000),
                       'X3': np.random.normal(size=1_000_000),
                       'X4': np.random.normal(size=1_000_000),
                       'X5': np.random.normal(size=1_000_000),
                       'X6': np.random.normal(size=1_000_000),
                       'X7': np.random.normal(size=1_000_000),
                       'X8': np.random.normal(size=1_000_000),
                       'X9': np.random.normal(size=1_000_000),
                       'product_group': np.random.choice(['total','produce'],size=1_000_000),
                       'modality': np.random.choice(['all','online'],size=1_000_000)})
    return (df,)


@app.cell
def _(FastLSTSQ):
    fu = FastLSTSQ(Y=["Y1","Y2","Y3"],
                    T="T",
                    G=["G1","G2","G3","G4","G5","G6","G7","G8"],
                    X=[f"X{i}" for i in range(1,10)],
                    engine='cpu')
    return (fu,)


@app.cell
def _(df, fu):
    fu.fit_and_estimate(data=df)
    return


@app.cell
def _(FastLSTSQ, df):
    import timeit

    cpu_times = []
    gpu_times = []

    fu1 = FastLSTSQ(Y=["Y1","Y2","Y3"], T="T", G=["G1","G2","G3","G4","G5","G6","G7","G8"], X=[f"X{i}" for i in range(1,10)], engine='cpu')
    fu1.fit_and_estimate(data=df)

    for _ in range(5):
        start_time = timeit.default_timer()
        fu2 = FastLSTSQ(Y=["Y1","Y2","Y3"], T="T", G=["G1","G2","G3","G4","G5","G6","G7","G8"], X=[f"X{i}" for i in range(1,10)], engine='cpu')
        fu2.fit_and_estimate(data=df)
        cpu_times.append(timeit.default_timer() - start_time)

    fu3 = FastLSTSQ(Y=["Y1","Y2","Y3"], T="T", G=["G1","G2","G3","G4","G5","G6","G7","G8"], X=[f"X{i}" for i in range(1,10)], engine='gpu')
    fu3.fit_and_estimate(data=df)

    for _ in range(5):
        start_time = timeit.default_timer()
        fu4 = FastLSTSQ(Y=["Y1","Y2","Y3"], T="T", G=["G1","G2","G3","G4","G5","G6","G7","G8"], X=[f"X{i}" for i in range(1,10)], engine='gpu')
        fu4.fit_and_estimate(data=df)
        gpu_times.append(timeit.default_timer() - start_time)

    print(f"CPU times: {cpu_times}")
    print(f"GPU times: {gpu_times}")
    print(f"Average CPU time: {sum(cpu_times)/len(cpu_times):.4f} seconds")
    print(f"Average GPU time: {sum(gpu_times)/len(gpu_times):.4f} seconds")
    return cpu_times, fu1, fu2, fu3, fu4, gpu_times, start_time, timeit


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
