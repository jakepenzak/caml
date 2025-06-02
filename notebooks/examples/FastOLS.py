import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # FastOLS

    In this notebook, we'll walk through an example of generating synthetic data, estimating treatment effects (ATEs, GATEs, and CATEs) using `FastOLS`, and comparing to our ground truth.

    `FastOLS` is particularly useful when efficiently estimating ATEs and GATEs is of primary interest and the treatment is exogenous or confounding takes on a particularly simple functional form.

    `FastOLS` assumes linear treatment effects & heterogeneity. This is generally sufficient for estimation of ATEs and GATEs, but can perform poorly in CATE estimation & prediction when heterogeneity is complex & nonlinear. For high quality CATE estimation, we recommend leveraging [CamlCATE](../04_Reference/CamlCATE.qmd).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Generate Synthetic Data

    Here we'll leverage the [`SyntheticDataGenerator`](../04_Reference/SyntheticDataGenerator.qmd) class to generate a linear synthetic data generating process, with an exogenous binary treatment, a continuous & a binary outcome, and binary & continuous mediating covariates.
    """
    )
    return


@app.cell
def _():
    from caml.logging import configure_logging
    import logging

    configure_logging(level=logging.DEBUG)
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data_generator = SyntheticDataGenerator(
        n_obs=10_000,
        n_cont_outcomes=1,
        n_binary_outcomes=1,
        n_binary_treatments=1,
        n_cont_confounders=2,
        n_cont_modifiers=3,
        n_binary_modifiers=2,
        stddev_outcome_noise=1,
        stddev_treatment_noise=1,
        causal_model_functional_form="linear",
        seed=10,
    )
    return (data_generator,)


@app.cell
def _(mo):
    mo.md(r"""We can print our simulated data via:""")
    return


@app.cell
def _(data_generator):
    data_generator.df
    return


@app.cell
def _(mo):
    mo.md(r"""To inspect our true data generating process, we can call `data_generator.dgp`. Furthermore, we will have our true CATEs and ATEs at our disposal via `data_generator.cates` & `data_generator.ates`, respectively. We'll use this as our source of truth for performance evaluation of our CATE estimator.""")
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
    mo.md(
        r"""
    ## Running FastOLS

    ### Class Instantiation

    We can instantiate and observe our `FastOLS` object via:
    """
    )
    return


@app.cell
def _(data_generator):
    from caml import FastOLS

    fo_obj = FastOLS(
        Y=[c for c in data_generator.df.columns if "Y" in c],
        T="T1_binary",
        G=[
            c
            for c in data_generator.df.columns
            if "X" in c and ("bin" in c or "dis" in c)
        ],
        X=[c for c in data_generator.df.columns if "X" in c and "cont" in c],
        W=[c for c in data_generator.df.columns if "W" in c],
        xformula="+ W1_continuous**2",
        engine="cpu",
        discrete_treatment=True,
    )
    return (fo_obj,)


@app.cell
def _(fo_obj):
    print(fo_obj)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Fitting OLS model

    We can now leverage the `fit` method to estimate the model outlined by `fo_obj.formula`. To capitalize on efficiency gains and parallelization in the estimation of GATEs, we will pass `estimate_effects=True`. The `n_jobs` argument will control the number of parallel jobs (GATE estimations) executed at a time. We will set `n_jobs=-1` to use all available cores for parallelization.

    ::: {.callout-warning}
    When dealing with large datasets, setting `n_jobs` to a more conservative value can help prevent OOM errors.
    :::

    For heteroskedasticity-robust variance estimation, we will also pass `robust_vcv=True`.
    """
    )
    return


@app.cell
def _(data_generator, fo_obj):
    fo_obj.fit(
        data_generator.df, n_jobs=-1, estimate_effects=True, robust_vcv=True
    )
    return


@app.cell
def _(mo):
    mo.md(r"""We can now inspect the model fitted results and estimated treatment effects:""")
    return


@app.cell
def _(fo_obj):
    fo_obj.params
    # fo_obj.vcv
    # fo_obj.std_err
    return


@app.cell
def _(fo_obj):
    fo_obj.treatment_effects.keys()
    return


@app.cell
def _(fo_obj):
    fo_obj.treatment_effects["overall"]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Here we have direct access to the model parameters (`fo_obj.params`), variance-covariance matrices (`fo_obj.vcv]`), standard_errors (`fo_obj.std_err`), and estimated treatment effects (`fo_obj.treatment_effects`).

    To make the treatment effect results more readable, we can leverage the `prettify_treatment_effects` method:
    """
    )
    return


@app.cell
def _(fo_obj):
    fo_obj.prettify_treatment_effects()
    return


@app.cell
def _(mo):
    mo.md(r"""Comparing our overall treatment effect (ATE) to the ground truth, we have:""")
    return


@app.cell
def _(data_generator):
    data_generator.ates
    return


@app.cell
def _(mo):
    mo.md("""We can also see what our GATEs are using `data_generator.cates`. Let's choose `X4_binary` in `1` group:""")
    return


@app.cell
def _(data_generator):
    data_generator.cates.iloc[
        data_generator.df.query("X4_binary == 1").index
    ].mean()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Custom Group Average Treatment Effects (GATEs)

    Let's now look at how we can estimate any arbitary GATE using `estimate_ate` method and prettify the results with `prettify_treatment_effects`.
    """
    )
    return


@app.cell
def _(data_generator, fo_obj):
    custom_gate_df = data_generator.df.query(
        "X4_binary == 1 & X2_continuous < -3"
    ).copy()

    custom_gate = fo_obj.estimate_ate(
        custom_gate_df,
        group="My Custom Group",
        membership="My Custom Membership",
        return_results_dict=True,
    )
    fo_obj.prettify_treatment_effects(effects=custom_gate)
    return (custom_gate_df,)


@app.cell
def _(mo):
    mo.md(r"""Let's compare this to the ground truth as well:""")
    return


@app.cell
def _(custom_gate_df, data_generator):
    data_generator.cates.iloc[custom_gate_df.index].mean()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Conditional Average Treatment Effects (CATEs)

    Let's now look at how we can estimate CATEs / approximate individual-level treatment effects via `estimate_cate` method

    ::: {.callout-note}
    The `predict` method is a simple alias for `estimate_cate`. Either can be used, but namespacing was created to higlight that `estimate_cate` / `predict` can be used for out of sample treatment effect prediction.
    :::
    """
    )
    return


@app.cell
def _(data_generator, fo_obj):
    cates = fo_obj.estimate_cate(data_generator.df)

    cates
    return


@app.cell
def _(mo):
    mo.md(r"""If we wanted additional information on CATEs (such as standard errors), we can call:""")
    return


@app.cell
def _(data_generator, fo_obj):
    fo_obj.estimate_cate(data_generator.df, return_results_dict=True)
    return


@app.cell
def _(mo):
    mo.md(r"""Now, let's make our cate predictions:""")
    return


@app.cell
def _(data_generator, fo_obj):
    cate_predictions = fo_obj.predict(data_generator.df)

    ## We can also make predictions of the outcomes, if desired.
    # fo_obj.predict(data_generator.df, mode="outcome")
    return (cate_predictions,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Let's now look at the Precision in Estimating Heterogeneous Effects (PEHE) (e.g., RMSE) and plot some results for the treatment effects on each outcome:

    #### Effect of *binary* T1 on *continuous* Y1
    """
    )
    return


@app.cell
def _():
    from sklearn.metrics import root_mean_squared_error
    from caml.extensions.plots import (
        cate_true_vs_estimated_plot,
        cate_histogram_plot,
        cate_line_plot,
    )
    return (
        cate_histogram_plot,
        cate_line_plot,
        cate_true_vs_estimated_plot,
        root_mean_squared_error,
    )


@app.cell
def _(cate_predictions, data_generator, root_mean_squared_error):
    true_cates1 = data_generator.cates.iloc[:, 0]
    predicted_cates1 = cate_predictions[:, 0]
    root_mean_squared_error(true_cates1, predicted_cates1)
    return predicted_cates1, true_cates1


@app.cell
def _(cate_true_vs_estimated_plot, predicted_cates1, true_cates1):
    cate_true_vs_estimated_plot(
        true_cates=true_cates1, estimated_cates=predicted_cates1
    )
    return


@app.cell
def _(cate_histogram_plot, predicted_cates1, true_cates1):
    cate_histogram_plot(true_cates=true_cates1, estimated_cates=predicted_cates1)
    return


@app.cell
def _(cate_line_plot, predicted_cates1, true_cates1):
    cate_line_plot(
        true_cates=true_cates1, estimated_cates=predicted_cates1, window=20
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Effect of *binary* T1 on *binary* Y2""")
    return


@app.cell
def _(cate_predictions, data_generator, root_mean_squared_error):
    true_cates2 = data_generator.cates.iloc[:, 1]
    predicted_cates2 = cate_predictions[:, 1]
    root_mean_squared_error(true_cates2, predicted_cates2)
    return predicted_cates2, true_cates2


@app.cell
def _(cate_true_vs_estimated_plot, predicted_cates2, true_cates2):
    cate_true_vs_estimated_plot(
        true_cates=true_cates2, estimated_cates=predicted_cates2
    )
    return


@app.cell
def _(cate_histogram_plot, predicted_cates2, true_cates2):
    cate_histogram_plot(true_cates=true_cates2, estimated_cates=predicted_cates2)
    return


@app.cell
def _(cate_line_plot, predicted_cates2, true_cates2):
    cate_line_plot(
        true_cates=true_cates2, estimated_cates=predicted_cates2, window=20
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ::: {.callout-note}
    The CATE estimates for binary outcome using simulated data may perform poorly b/c of non-linear transformation (sigmoid) of linear logodds. In general, `FastOLS` should be prioritized when ATEs and GATEs are of primary interest. For high quality CATE estimation, we recommend leveraging [CamlCATE](../04_Reference/CamlCATE.qmd).
    :::
    """
    )
    return


if __name__ == "__main__":
    app.run()
