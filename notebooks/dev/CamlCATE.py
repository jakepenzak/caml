import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# AutoCate API Usage""")
    return


@app.cell
def _(mo):
    mo.md("""## Synthetic Data""")
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data = SyntheticDataGenerator(
        n_obs=1_000,
        n_cont_outcomes=1,
        n_binary_outcomes=0,
        n_binary_treatments=1,
        n_cont_confounders=0,
        n_binary_confounders=0,
        n_discrete_confounders=0,
        n_cont_modifiers=3,
        n_binary_modifiers=3,
        n_discrete_modifiers=2,
        stddev_outcome_noise=1,
        stddev_treatment_noise=1,
        causal_model_functional_form="linear",
        seed=20,
    )


    synthetic_df = data.df
    cate_df = data.cates
    ate_df = data.ates
    dgp = data.dgp
    return ate_df, cate_df, dgp, synthetic_df


@app.cell
def _(synthetic_df):
    synthetic_df
    return


@app.cell
def _(dgp):
    dgp
    return


@app.cell
def _(cate_df):
    cate_df
    return


@app.cell
def _(ate_df):
    ate_df
    return


@app.cell
def _(synthetic_df):
    for col in synthetic_df.columns:
        if "discrete" in col:
            synthetic_df[col] = synthetic_df[col].astype("category")
    return


@app.cell
def _(mo):
    mo.md("""## Core API""")
    return


@app.cell
def _(mo):
    mo.md("""### AutoCATE""")
    return


@app.cell
def _(mo):
    mo.md("""#### Class Instantiation""")
    return


@app.cell
def _(synthetic_df):
    from caml import AutoCATE

    outcome = [c for c in synthetic_df.columns if "Y" in c][0]
    treatment = [c for c in synthetic_df.columns if "T" in c][0]

    caml = AutoCATE(Y=outcome,
                    T=treatment,
                    X=[c for c in synthetic_df.columns if 'X' in c],
                    W=[c for c in synthetic_df.columns if 'W' in c],
                    discrete_treatment=True if "binary" in treatment or "discrete" in treatment else False,
                    discrete_outcome=True if "binary" in outcome else False,
                    model_Y={"time_budget": 5},
                    model_T={"time_budget": 5},
                    model_regression={"time_budget": 5},
                    n_jobs=-1,
                    use_ray=False,
                    ray_remote_func_options_kwargs=None,
                    use_spark=False,
                    seed=None)


    return (caml,)


@app.cell
def _(mo):
    mo.md("""#### Nuissance Function AutoML""")
    return


@app.cell
def _(caml, synthetic_df):
    final_estimator = caml.fit(synthetic_df)
    return


@app.cell
def _(mo):
    mo.md(r"""# Legacy""")
    return


@app.cell
def _(caml):
    caml.auto_nuisance_functions(
        flaml_Y_kwargs={
            "time_budget": 120,
            "estimator_list": [
                "lgbm",
                "rf",
                "extra_tree",
                "xgb_limitdepth",
                "lrl1",
                "lrl2",
            ],
        },
        flaml_T_kwargs={
            "time_budget": 120,
            "estimator_list": [
                "lgbm",
                "rf",
                "extra_tree",
                "xgb_limitdepth",
                "lrl1",
                "lrl2",
            ],
        },
        use_ray=False,
        use_spark=False,
    )
    return


@app.cell
def _(mo):
    mo.md("""#### Fit and ensemble CATE models""")
    return


@app.cell
def _(caml):
    caml.fit_validator(
        cate_estimators=[
            "LinearDML",
            "CausalForestDML",
            "NonParamDML",
            "SparseLinearDML-2D",
            "DRLearner",
            "ForestDRLearner",
            "LinearDRLearner",
            "DomainAdaptationLearner",
            "SLearner",
            "TLearner",
            "XLearner",
        ],
        additional_cate_estimators=[],
        rscorer_kwargs={},
        use_ray=False,
        ray_remote_func_options_kwargs={},
        ensemble=False,
        validation_size=0.2,
        test_size=0.2,
        sample_size=1.0,
        n_jobs=-1,
    )
    return


@app.cell
def _(caml):
    caml.rscores
    return


@app.cell
def _(mo):
    mo.md("""#### CATE Validation""")
    return


@app.cell
def _(caml):
    import matplotlib.pyplot as plt

    caml.validate(n_groups=4, n_bootstrap=100, print_full_report=True)

    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""#### Refit best estimator on full dataset""")
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
    mo.md("""#### Predict CATEs""")
    return


@app.cell
def _(caml):
    ## "Out of sample" predictions

    cate_predictions = caml.predict(T0=0, T1=1)

    cate_predictions
    return (cate_predictions,)


@app.cell
def _():
    #### Summarize CATEs
    return


@app.cell
def _(caml):
    cate_summary = caml.summarize()

    cate_summary
    return


@app.cell
def _(cate_df):
    cate_df.describe()
    return


@app.cell
def _(mo):
    mo.md("""#### Access my dataframe, estimator object, and get string representation of class""")
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
            print(caml.input_names)
    else:
        print(final_estimator)
        print(caml.input_names)
    return


@app.cell
def _(mo):
    mo.md("""# Plots""")
    return


@app.cell
def _(cate_df, cate_predictions, synthetic_df):
    from caml.extensions.plots import (
        cate_histogram_plot,
        cate_true_vs_estimated_plot,
        cate_line_plot,
    )

    synthetic_df["cate_predictions"] = cate_predictions
    synthetic_df["true_cates"] = cate_df.iloc[:, 0]

    lower = synthetic_df["true_cates"].quantile(0.05)
    upper = synthetic_df["true_cates"].quantile(0.95)
    synthetic_df_trimmed = synthetic_df[
        (synthetic_df["true_cates"] >= lower)
        & (synthetic_df["true_cates"] <= upper)
    ]
    return (
        cate_histogram_plot,
        cate_line_plot,
        cate_true_vs_estimated_plot,
        synthetic_df_trimmed,
    )


@app.cell
def _(cate_true_vs_estimated_plot, synthetic_df_trimmed):
    cate_true_vs_estimated_plot(
        true_cates=synthetic_df_trimmed["true_cates"],
        estimated_cates=synthetic_df_trimmed["cate_predictions"],
    )
    return


@app.cell
def _(cate_histogram_plot, synthetic_df):
    cate_histogram_plot(estimated_cates=synthetic_df["cate_predictions"])
    return


@app.cell
def _(cate_histogram_plot, synthetic_df):
    cate_histogram_plot(
        estimated_cates=synthetic_df["cate_predictions"],
        true_cates=synthetic_df["true_cates"],
    )
    return


@app.cell
def _(cate_line_plot, synthetic_df):
    cate_line_plot(estimated_cates=synthetic_df["cate_predictions"], window=30)
    return


@app.cell
def _(cate_line_plot, synthetic_df):
    cate_line_plot(
        estimated_cates=synthetic_df["cate_predictions"],
        true_cates=synthetic_df["true_cates"],
        window=20,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
