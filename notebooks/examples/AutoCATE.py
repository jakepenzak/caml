import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # AutoCATE

    In this notebook, we'll walk through an example of generating synthetic data, running `AutoCATE`, and visualizing results using the ground truth as reference.

    `AutoCATE` is particularly useful when highly accurate CATE estimation is of primary interest in the presence of exogenous treatment, simple linear confounding, or complex non-linear confounding.

    `AutoCATE` enables the use of various CATE models with varying assumptions on functional form of treatment effects & heterogeneity. When a set of CATE models are considered, the final CATE model is automatically selected based on validation set performance.

    > 🚀**Forthcoming:** The `AutoCATE` class is in a very experimental, infancy stage. Additional scoring techniques & a more robust AutoML framework for CATE estimators is on our roadmap.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Generate Synthetic Data

    Here we'll leverage the [`SyntheticDataGenerator`](../04_Reference/SyntheticDataGenerator.qmd) class to generate a linear synthetic data generating process, with a binary treatment, continuous outcome, and a mix of confounding/mediating continuous covariates.
    """)
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data_generator = SyntheticDataGenerator(
        n_obs=10_000,
        n_cont_outcomes=1,
        n_binary_treatments=1,
        n_cont_confounders=2,
        n_cont_modifiers=2,
        n_confounding_modifiers=1,
        causal_model_functional_form="linear",
        seed=10,
    )
    return (data_generator,)


@app.cell
def _(mo):
    mo.md(r"""
    We can print our simulated data via:
    """)
    return


@app.cell
def _(data_generator):
    data_generator.df
    return


@app.cell
def _(mo):
    mo.md(r"""
    To inspect our true data generating process, we can call `data_generator.dgp`. Furthermore, we will have our true CATEs and ATEs at our disposal via `data_generator.cates` & `data_generator.ates`, respectively. We'll use this as our source of truth for performance evaluation of our CATE estimator.
    """)
    return


@app.cell
def _(data_generator):
    for t, df in data_generator.dgp.items():
        print(f"\nDGP for {t}:")
        print(df)
    return


@app.cell
def _(data_generator):
    data_generator.cates
    return


@app.cell
def _(data_generator):
    data_generator.ates
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Running AutoCATE
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Class Instantiation

    We can instantiate and observe our `AutoCATE` object via:

    ::: {.callout-tip}
    `W` can be leveraged if we want to use certain covariates only in our nuisance functions to control for confounding and not in the final CATE estimator. This can be useful if a confounder may be required to include, but for compliance reasons, we don't want our CATE model to leverage this feature (e.g., gender). However, this will restrict our available CATE estimators to orthogonal learners, since metalearners necessarily include all covariates. If you don't care about `W` being in the final CATE estimator, pass it as `X`, as done below.
    :::
    """)
    return


@app.cell
def _(data_generator):
    from caml import AutoCATE

    auto_cate = AutoCATE(
        Y="Y1_continuous",
        T="T1_binary",
        X=[c for c in data_generator.df.columns if "X" in c]
        + [c for c in data_generator.df.columns if "W" in c],
        discrete_treatment=True,
        discrete_outcome=False,
        model_Y={
            "time_budget": 10,
            "estimator_list": ["rf", "extra_tree", "xgb_limitdepth"],
        },
        model_T={
            "time_budget": 10,
            "estimator_list": ["rf", "extra_tree", "xgb_limitdepth"],
        },
        model_regression={
            "time_budget": 10,
            "estimator_list": ["rf", "extra_tree", "xgb_limitdepth"],
        },
        enable_categorical=True,
        n_jobs=-1,
        use_ray=False,
        ray_remote_func_options_kwargs=None,
        use_spark=False,
        seed=None,
    )
    return (auto_cate,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Fitting End-to-End AutoCATE pipeline
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    First, I can inspect available estimators out of the box:
    """)
    return


@app.cell
def _(auto_cate):
    auto_cate.available_estimators
    return


@app.cell
def _(mo):
    mo.md(r"""
    And create another one if desired:
    """)
    return


@app.cell
def _():
    from caml import AutoCateEstimator
    from econml.dml import LinearDML

    my_custom_estimator = AutoCateEstimator(name="MyCustomEstimator",estimator=LinearDML())
    return (my_custom_estimator,)


@app.cell
def _(auto_cate, data_generator, my_custom_estimator):
    auto_cate.fit(data_generator.df, cate_estimators=['LinearDML','CausalForestDML','TLearner'], additional_cate_estimators=[my_custom_estimator])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ::: {.callout-note}
    The selected CATE model defaults to the one with the highest [RScore](https://econml.azurewebsites.net/_autosummary/econml.score.RScorer.html#econml.score.RScorer).
    :::
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Validating Results with Ground Truth
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Average Treatment Effect (ATE)

    We'll use the `summarize()` method after obtaining our predictions above, where our the displayed mean represents our Average Treatment Effect (ATE).
    """)
    return


@app.cell
def _(auto_cate, data_generator):
    auto_cate.estimate_ate(data_generator.df[auto_cate.X])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now comparing this to our ground truth, we see the model performed well the true ATE:
    """)
    return


@app.cell
def _(data_generator):
    data_generator.ates
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Conditional Average Treatment Effect (CATE)

    Now we want to see how the estimator performed in modeling the true CATEs.

    First, we can simply compute the Precision in Estimating Heterogeneous Effects (PEHE), which is simply the Root Mean Squared Error (RMSE):
    """)
    return


@app.cell
def _(auto_cate, data_generator):
    from sklearn.metrics import root_mean_squared_error

    cate_predictions = auto_cate.estimate_cate(
        data_generator.df[auto_cate.X]
    )  # `predict` and `effect` are aliases
    true_cates = data_generator.cates.to_numpy()
    root_mean_squared_error(true_cates, cate_predictions)
    return cate_predictions, true_cates


@app.cell
def _(mo):
    mo.md(r"""
    Not bad! Now let's use some visualization techniques:
    """)
    return


@app.cell
def _(cate_predictions, true_cates):
    from caml.extensions.plots import cate_true_vs_estimated_plot

    cate_true_vs_estimated_plot(
        true_cates=true_cates, estimated_cates=cate_predictions
    )
    return


@app.cell
def _(cate_predictions, true_cates):
    from caml.extensions.plots import cate_histogram_plot

    cate_histogram_plot(true_cates=true_cates, estimated_cates=cate_predictions)
    return


@app.cell
def _(cate_predictions, true_cates):
    from caml.extensions.plots import cate_line_plot

    cate_line_plot(
        true_cates=true_cates.flatten(),
        estimated_cates=cate_predictions.flatten(),
        window=20,
    )
    return (cate_line_plot,)


@app.cell
def _(mo):
    mo.md(r"""
    Overall, we can see the model performed remarkably well!

    Now, we can also get standard errors by obtaining the [EconML Inference Results](https://www.pywhy.org/EconML/reference.html#inference) by passing `return_inference=True` to `estimate_cate` (or `predict` or `effect`):
    """)
    return


@app.cell
def _(auto_cate, cate_line_plot, cate_predictions, data_generator, true_cates):
    inference = auto_cate.estimate_cate(
        data_generator.df[auto_cate.X], return_inference=True
    )
    stderrs = inference.stderr

    cate_line_plot(
        true_cates=true_cates.flatten(),
        estimated_cates=cate_predictions.flatten(),
        window=20,
        standard_errors=stderrs,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
