import marimo

__generated_with = "0.10.18"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # CamlCATE API Usage

        Here we'll walk through an example of generating synthetic data, running CamlCATE, and visualizing results using the ground truth as reference.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Generate Synthetic Data

        Here we'll leverage the [`CamlSyntheticDataGenerator`](../04_Reference/CamlSyntheticDataGenerator.qmd) class to generate a linear synthetic data generating process, with a binary treatment, continuous outcome, and a mix of confounding/mediating continuous covariates.
        """
    )
    return


@app.cell
def _():
    from caml.extensions.synthetic_data import CamlSyntheticDataGenerator

    data =  CamlSyntheticDataGenerator(n_obs=10_000,
                                      n_cont_outcomes=1,
                                      n_binary_treatments=1,
                                      n_cont_confounders=2,
                                      n_cont_modifiers=2,
                                      n_confounding_modifiers=1,
                                      causal_model_functional_form="linear",
                                      n_nonlinear_transformations=5,
                                      n_nonlinear_interactions=2,
                                      seed=1)
    return CamlSyntheticDataGenerator, data


@app.cell
def _(mo):
    mo.md(r"""We can print our simulated data via:""")
    return


@app.cell
def _(data):
    data.df
    return


@app.cell
def _(mo):
    mo.md(r"""To inspect our true data generating process, we can call `data.dgp`. Furthermore, we will have our true CATEs and ATEs at our disposal via `data.cates` & `data.ates`, respectively. We'll use this as our source of truth for performance evaluation of our CATE estimator.""")
    return


@app.cell
def _(data):
    for t, df in data.dgp.items():
        print(f"\nDGP for {t}:")
        print(df)
    return df, t


@app.cell
def _(data):
    data.cates
    return


@app.cell
def _(data):
    data.ates
    return


@app.cell
def _(mo):
    mo.md(r"""## Running CamlCATE""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Class Instantiation

        We can instantiate and observe our CamlCATE object via:

        > 💡 **Tip:** `W` can be leveraged if we want to use certain covariates only in our nuisance functions to control for confounding and not in the final CATE estimator. This can be useful if a confounder may be required to include, but for compliance reasons, we don't want our CATE model to leverage this feature (e.g., gender). However, this will restrict our available CATE estimators to orthogonal learners, since metalearners necessarily include all covariates. If you don't care about `W` being in the final CATE estimator, pass it as `X`, as done below.
        """
    )
    return


@app.cell
def _(data):
    from caml import CamlCATE

    caml_obj = CamlCATE(df=data.df,
                    Y="Y1_continuous",
                    T="T1_binary",
                    X=[c for c in data.df.columns if 'X' in c]
                        + [c for c in data.df.columns if 'W' in c],
                    discrete_treatment=True,
                    discrete_outcome=False,
                    verbose=1)
    return CamlCATE, caml_obj


@app.cell
def _(caml_obj):
    print(caml_obj)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Nuisance Function AutoML

        We can then obtain our nuisance functions / regression & propensity models via Flaml AutoML:
        """
    )
    return


@app.cell
def _(caml_obj):
    caml_obj.auto_nuisance_functions(
        flaml_Y_kwargs={"time_budget": 30,
                        "verbose":0,
                        "estimator_list":["rf", "extra_tree", "xgb_limitdepth"]},
        flaml_T_kwargs={"time_budget": 30,
                        "verbose":0,
                        "estimator_list":["rf", "extra_tree", "xgb_limitdepth"]},
    )
    return


@app.cell
def _(caml_obj):
    print(caml_obj.model_Y_X_W)
    print(caml_obj.model_Y_X_W_T)
    print(caml_obj.model_T_X_W)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Fit CATE Estimators

        Now that we have obtained our first-stage models, we can fit our CATE estimators via:

        > 📝 **Note:** The selected model defaults to the one with the highest [RScore](https://econml.azurewebsites.net/_autosummary/econml.score.RScorer.html#econml.score.RScorer). All fitted models are still accessible via the `cate_estimators` attribute and if you want to change default estimator, you can run `caml_obj._validation_estimator = {different_model}`.

        > 🚀**Forthcoming:** Additional scoring techniques & AutoML for CATE estimators is on our roadmap.
        """
    )
    return


@app.cell
def _(caml_obj):
    caml_obj.fit_validator(
        cate_estimators=[
            "LinearDML",
            "CausalForestDML",
            "ForestDRLearner",
            "LinearDRLearner",
            "DomainAdaptationLearner",
            "SLearner",
            "TLearner",
            "XLearner",
        ],
        validation_size=0.2,
        test_size=0.2,
        n_jobs=-1,
    )
    return


@app.cell
def _(caml_obj):
    caml_obj.validation_estimator
    return


@app.cell
def _(caml_obj):
    caml_obj.cate_estimators
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Validate model on test hold out set

        Here we can validate our model on the test hold out set. Currently, this is only available for when continuous outcomes and binary treatments exist.
        """
    )
    return


@app.cell
def _(caml_obj):
    caml_obj.validate()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Refit our selected model on the entire dataset

        Now that we have selected our top performer and validated results on the test set, we can fit our final model on the entire dataset.
        """
    )
    return


@app.cell
def _(caml_obj):
    caml_obj.fit_final()
    return


@app.cell
def _(caml_obj):
    caml_obj.final_estimator
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Validating Results with Ground Truth

        First, we will obtain our predictions.
        """
    )
    return


@app.cell
def _(caml_obj):
    cate_predictions = caml_obj.predict()
    return (cate_predictions,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Average Treatment Effect (ATE)

        We'll use the `summarize()` method after obtaining our predictions above, where our the displayed mean represents our Average Treatment Effect (ATE).
        """
    )
    return


@app.cell
def _(caml_obj):
    caml_obj.summarize()
    return


@app.cell
def _(mo):
    mo.md(r"""Now comparing this to our ground truth, we see the model performed well the true ATE:""")
    return


@app.cell
def _(data):
    data.ates
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Conditional Average Treatment Effect (CATE)

        Now we want to see how the estimator performed in modeling the true CATEs.

        First, we can simply compute the Precision in Estimating Heterogeneous Effects (PEHE), which is simply the Mean Squared Error (MSE):
        """
    )
    return


@app.cell
def _(cate_predictions, data):
    from sklearn.metrics import mean_squared_error

    true_cates = data.cates.iloc[:, 0]
    mean_squared_error(true_cates,cate_predictions)
    return mean_squared_error, true_cates


@app.cell
def _(mo):
    mo.md(r"""Not bad! Now let's use some visualization techniques:""")
    return


@app.cell
def _(cate_predictions, true_cates):
    from caml.extensions.plots import cate_true_vs_estimated_plot

    cate_true_vs_estimated_plot(true_cates=true_cates, estimated_cates=cate_predictions)
    return (cate_true_vs_estimated_plot,)


@app.cell
def _(cate_predictions, true_cates):
    from caml.extensions.plots import cate_histogram_plot

    cate_histogram_plot(true_cates=true_cates, estimated_cates=cate_predictions)
    return (cate_histogram_plot,)


@app.cell
def _(cate_predictions, true_cates):
    from caml.extensions.plots import cate_line_plot

    cate_line_plot(true_cates=true_cates, estimated_cates=cate_predictions, window=20)
    return (cate_line_plot,)


@app.cell
def _(mo):
    mo.md(r"""Overall, we can see the model performed remarkably well!~""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Obtaining Model Objects & Artifacts for Production Systems

        In many production settings, we will want to store our model, information on the features used, etc. We provide attributes that to pull key information (more to be added later as class evolves)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Grabbing final model object:""")
    return


@app.cell
def _(caml_obj):
    caml_obj.final_estimator
    return


@app.cell
def _(mo):
    mo.md(r"""Grabbing input features:""")
    return


@app.cell
def _(caml_obj):
    caml_obj.input_names
    return


@app.cell
def _(mo):
    mo.md(r"""Grabbing all fitted CATE estimators:""")
    return


@app.cell
def _(caml_obj):
    caml_obj.cate_estimators
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
