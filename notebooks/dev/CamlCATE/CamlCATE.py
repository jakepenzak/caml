import marimo

__generated_with = "0.10.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Caml API Usage""")
    return


@app.cell
def _():
    import os
    import sys

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    datasets = [
        "partially_linear_simple",
        "fully_heterogenous",
        "partially_linear_constant",
        "dowhy_linear",
    ]
    backends = ["pandas", "pyspark", "polars"]

    df_backend = backends[0]
    dataset = datasets[3]
    return backends, dataset, datasets, df_backend, os, sys


@app.cell
def _(mo):
    mo.md(r"""## Synthetic Data""")
    return


@app.cell
def _(dataset):
    from caml.extensions.synthetic_data import (
        make_partially_linear_dataset_simple,
        make_fully_heterogeneous_dataset,
        make_partially_linear_dataset_constant,
        make_dowhy_linear_dataset,
    )

    if dataset == "partially_linear_simple":
        df, true_cates, true_ate = make_partially_linear_dataset_simple(
            n_obs=5000,
            n_confounders=5,
            dim_heterogeneity=2,
            binary_treatment=True,
            seed=None,
        )
        df["true_cates"] = true_cates
    elif dataset == "fully_heterogenous":
        df, true_cates, true_ate = make_fully_heterogeneous_dataset(
            n_obs=10_000,
            n_confounders=10,
            theta=4.0,
            seed=None,
        )
        df["true_cates"] = true_cates
    elif dataset == "partially_linear_constant":
        df, true_cates, true_ate = make_partially_linear_dataset_constant(
            n_obs=5000,
            ate=4.0,
            n_confounders=5,
            dgp="make_plr_CCDDHNR2018",  # make_plr_turrell2018
            seed=None,
        )
        df["true_cates"] = true_cates
    elif dataset == "dowhy_linear":
        df, true_cates, true_ate = make_dowhy_linear_dataset(
            beta=2.0,
            n_obs=10_000,
            n_confounders=10,
            n_discrete_confounders=3,
            n_effect_modifiers=10,
            n_discrete_effect_modifiers=3,
            n_treatments=1,
            binary_treatment=True,
            categorical_treatment=False,
            binary_outcome=False,
            seed=12,
        )

        for i in range(1, len(true_cates) + 1):
            if isinstance(true_cates[f"d{i}"], list):
                df[f"true_cate_d{i}_1"] = true_cates[f"d{i}"][0]
                df[f"true_cate_d{i}_2"] = true_cates[f"d{i}"][1]
            else:
                df[f"true_cate_d{i}"] = true_cates[f"d{i}"]
    return (
        df,
        i,
        make_dowhy_linear_dataset,
        make_fully_heterogeneous_dataset,
        make_partially_linear_dataset_constant,
        make_partially_linear_dataset_simple,
        true_ate,
        true_cates,
    )


@app.cell
def _(df, df_backend):
    try:
        import polars as pl
        from pyspark.sql import SparkSession
    except ImportError:
        pass
    if df_backend == 'polars':
        df_pl = pl.from_pandas(df)
        spark = None
    elif df_backend == 'pandas':
        spark = None
        pass
    elif df_backend == 'pyspark':
        spark = SparkSession.builder\
        .master('local[1]').appName('local-tests')\
        .config('spark.executor.cores', '1')\
        .config('spark.executor.instances', '1')\
        .config('spark.sql.shuffle.partitions', '1')\
        .getOrCreate()

        df_spark = spark.createDataFrame(df)
    return SparkSession, df_pl, df_spark, pl, spark


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import CamlSyntheticDataGenerator

    data = CamlSyntheticDataGenerator(n_obs=10000,
                                      n_cont_outcomes=1,
                                      n_binary_outcomes=0,
                                      n_cont_treatments=1,
                                      n_binary_treatments=0,
                                      n_discrete_treatments=0,
                                      n_cont_confounders=2,
                                      n_binary_confounders=2,
                                      n_discrete_confounders=0,
                                      n_cont_heterogeneity_covariates=2,
                                      n_binary_heterogeneity_covariates=2,
                                      n_discrete_heterogeneity_covariates=0,
                                      n_heterogeneity_confounders=0,
                                      stddev_outcome_noise=3,
                                      stddev_treatment_noise=3,
                                      causal_model_functional_form='fully_linear',
                                      n_nonlinear_transformations=10,
                                      n_nonlinear_interactions=5,
                                      seed=10)

    synthetic_df = data.df
    cate_df = data.cates
    ate_df = data.ates
    dgp = data.dgp
    return (
        CamlSyntheticDataGenerator,
        ate_df,
        cate_df,
        data,
        dgp,
        synthetic_df,
    )


@app.cell
def _(cate_df):
    cate_df
    return


@app.cell
def _(cate_df):
    cate_df
    return


@app.cell
def _(synthetic_df):
    synthetic_df
    return


@app.cell
def _(ate_df):
    ate_df
    return


@app.cell
def _(dgp):
    dgp
    return


@app.cell(hide_code=True)
def _():
    # import numpy as np
    # def _sigmoid(x):
    #     """Numerically stable sigmoid"""

    #     result = np.zeros_like(x, dtype=float)

    #     pos_mask = x >= 0
    #     neg_mask = ~pos_mask

    #     result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    #     result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

    #     return result

    # d_sigmoid = lambda x: _sigmoid(x)*(1-_sigmoid(x))

    # synthetic_df['int_T1_continuous_X1_continuous'] = synthetic_df["T1_continuous"] * synthetic_df["X1_continuous"]

    # params = np.array(dgp["Y1_binary"]["params"])
    # values = np.array(synthetic_df[[c for c in synthetic_df.columns if "W" in c or "X" in c or "T" in c]])

    # cate_df['manual_estimates'] = (-2.5737803491434144-0.7576456462998822*synthetic_df['X1_continuous'])*d_sigmoid(values @ params)
    return


@app.cell
def _(cate_df):
    cate_df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""## Core API""")
    return


@app.cell
def _(mo):
    mo.md(r"""### CamlCATE""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Class Instantiation""")
    return


@app.cell
def _(synthetic_df):
    from caml import CamlCATE

    caml = CamlCATE(df=synthetic_df,
                    Y='Y1_binary',
                    T='T1_binary',
                    X=[c for c in synthetic_df.columns if 'X' in c or 'W' in c],
                    W=[],
                    discrete_treatment=True,
                    discrete_outcome=True,
                    seed=10,
                    verbose=1)
    return CamlCATE, caml


@app.cell
def _(caml):
    print(caml)
    return


@app.cell
def _(caml):
    caml.df
    return


@app.cell
def _(mo):
    mo.md(r"""#### Nuissance Function AutoML""")
    return


@app.cell
def _(caml):
    caml.auto_nuisance_functions(
        flaml_Y_kwargs={"time_budget": 20},
        flaml_T_kwargs={"time_budget": 20},
        use_ray=False,
        use_spark=False,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Fit and ensemble CATE models""")
    return


@app.cell
def _(caml):
    caml.fit_validator(
        subset_cate_models=[
            "LinearDML",
            "CausalForestDML",
            "NonParamDML",
            "AutoNonParamDML",
            "SparseLinearDML-2D",
            "DRLearner"
            "AutoDRLearner",
            "ForestDRLearner",
            "LinearDRLearner",
            "SparseLinearDRLearner-2D",
            "DomainAdaptationLearner",
            "SLearner",
            "TLearner",
            "XLearner",
        ],
        rscorer_kwargs={},
        use_ray=False,
        ray_remote_func_options_kwargs={},
        sample_fraction=1.0,
        n_jobs=-1,
    )
    return


@app.cell
def _(caml):
    caml.validation_estimator
    return


@app.cell
def _(mo):
    mo.md(r"""#### CATE Validation""")
    return


@app.cell
def _(caml):
    caml.validate(n_groups=4,n_bootstrap=100,print_full_report=True)
    return


@app.cell
def _(mo):
    mo.md(r"""#### Refit best estimator on full dataset""")
    return


@app.cell
def _(caml):
    caml.fit_final()
    return


@app.cell
def _(caml):
    caml.final_estimator
    return


@app.cell
def _(mo):
    mo.md(r"""#### Predict CATEs""")
    return


@app.cell
def _(caml):
    ## "Out of sample" predictions

    cate_predictions = caml.predict()

    cate_predictions
    return (cate_predictions,)


@app.cell
def _(caml):
    caml._cate_predictions['cate_predictions_0_1'] = caml._cate_predictions['cate_predictions_0_1'].ravel()
    return


@app.cell
def _():
    #### Summarize CATEs
    return


@app.cell
def _(caml):
    cate_summary = caml.summarize()

    cate_summary
    return (cate_summary,)


@app.cell
def _(ate_df):
    ate_df
    return


@app.cell
def _(mo):
    mo.md(r"""#### Access my dataframe, estimator object, and get string representation of class""")
    return


@app.cell
def _(caml):
    caml.df
    return


@app.cell
def _(caml):
    from econml.score import EnsembleCateEstimator

    # Use this estimator object as pickled object for optimized inference
    final_estimator = caml.final_estimator

    if isinstance(final_estimator, EnsembleCateEstimator):
        for mod in final_estimator._cate_models:
            print(mod)
            print(mod._input_names)
    else:
        print(final_estimator)
        print(final_estimator._input_names)
    return EnsembleCateEstimator, final_estimator, mod


@app.cell
def _(mo):
    mo.md(r"""# Plots""")
    return


@app.cell
def _(cate_df, cate_predictions, synthetic_df):
    from caml.extensions.plots import cate_histogram_plot, cate_true_vs_estimated_plot, cate_line_plot
    synthetic_df['cate_predictions'] = cate_predictions
    synthetic_df['true_cates'] = cate_df.iloc[:, 0]
    return cate_histogram_plot, cate_line_plot, cate_true_vs_estimated_plot


@app.cell
def _(cate_true_vs_estimated_plot, synthetic_df):
    cate_true_vs_estimated_plot(true_cates=synthetic_df['true_cates'], estimated_cates=synthetic_df['cate_predictions'])
    return


@app.cell
def _(cate_histogram_plot, synthetic_df):
    cate_histogram_plot(estimated_cates=synthetic_df['cate_predictions'])
    return


@app.cell
def _(cate_histogram_plot, synthetic_df):
    cate_histogram_plot(estimated_cates=synthetic_df['cate_predictions'], true_cates=synthetic_df['true_cates'])
    return


@app.cell
def _(cate_line_plot, synthetic_df):
    cate_line_plot(estimated_cates=synthetic_df['cate_predictions'], window=30)
    return


@app.cell
def _(cate_line_plot, synthetic_df):
    cate_line_plot(estimated_cates=synthetic_df['cate_predictions'], true_cates=synthetic_df['true_cates'], window=20)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
