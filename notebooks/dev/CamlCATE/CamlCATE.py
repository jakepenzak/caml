# %% [markdown]
# # Caml API Usage

# %%
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

# %% [markdown]
# ## Create Synthetic Data

# %%
from caml.extensions.synthetic_data import (
    make_dowhy_linear_dataset,
    make_fully_heterogeneous_dataset,
    make_partially_linear_dataset_constant,
    make_partially_linear_dataset_simple,
)

if dataset == "partially_linear_simple":
    df, true_cates, true_ate = make_partially_linear_dataset_simple(
        n_obs=1000,
        n_confounders=5,
        dim_heterogeneity=2,
        binary_treatment=True,
        seed=None,
    )
    df["true_cates"] = true_cates
elif dataset == "fully_heterogenous":
    df, true_cates, true_ate = make_fully_heterogeneous_dataset(
        n_obs=1000,
        n_confounders=10,
        theta=4.0,
        seed=None,
    )
    df["true_cates"] = true_cates
elif dataset == "partially_linear_constant":
    df, true_cates, true_ate = make_partially_linear_dataset_constant(
        n_obs=1000,
        ate=4.0,
        n_confounders=5,
        dgp="make_plr_CCDDHNR2018",  # make_plr_turrell2018
        seed=None,
    )
    df["true_cates"] = true_cates
elif dataset == "dowhy_linear":
    df, true_cates, true_ate = make_dowhy_linear_dataset(
        beta=2.0,
        n_obs=10000,
        n_confounders=10,
        n_discrete_confounders=2,
        n_effect_modifiers=6,
        n_discrete_effect_modifiers=2,
        n_treatments=1,
        binary_treatment=True,
        categorical_treatment=False,
        binary_outcome=False,
        seed=0,
    )

    for i in range(1, len(true_cates) + 1):
        if isinstance(true_cates[f"d{i}"], list):
            df[f"true_cate_d{i}_1"] = true_cates[f"d{i}"][0]
            df[f"true_cate_d{i}_2"] = true_cates[f"d{i}"][1]
        else:
            df[f"true_cate_d{i}"] = true_cates[f"d{i}"]


df["uuid"] = df.index

# %%
import polars as pl

try:
    from pyspark.sql import SparkSession
except ImportError:
    pass

if df_backend == "polars":
    df = pl.from_pandas(df)
    spark = None
elif df_backend == "pandas":
    spark = None
    pass
elif df_backend == "pyspark":
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    df = spark.createDataFrame(df)

# %%
df

# %% [markdown]
# ## Core API

# %% [markdown]
# ### CamlCATE

# %% [markdown]
# #### Class Instantiation

# %%
from caml import CamlCATE

caml = CamlCATE(
    df=df,
    Y="y",
    T="d1",
    X=[c for c in df.columns if "X" in c],
    W=[c for c in df.columns if "W" in c],
    uuid="uuid",
    discrete_treatment=True,
    discrete_outcome=False,
    seed=0,
)

# %% [markdown]
# #### Nuissance Function AutoML

# %%
caml.auto_nuisance_functions(
    flaml_Y_kwargs={"time_budget": 60},
    flaml_T_kwargs={"time_budget": 60},
    use_ray=False,
    use_spark=False,
)

# %% [markdown]
# #### Fit and ensemble CATE models

# %%
caml.fit_validator(
    subset_cate_models=[
        "LinearDML",
        "NonParamDML",
        "DML-Lasso3d",
        "CausalForestDML",
        "XLearner",
        "DomainAdaptationLearner",
        "SLearner",
        "TLearner",
        "DRLearner",
    ],
    rscorer_kwargs={},
    use_ray=False,
    ray_remote_func_options_kwargs={},
)

# %%
caml.validation_estimator

# %% [markdown]
# #### CATE Validation

# %%
validation_results = caml.validate(estimator=None, print_full_report=True)

# %% [markdown]
# #### Refit best estimator on full dataset

# %%
caml.fit_final()

# %%
caml.final_estimator

# %% [markdown]
# #### Predict CATEs

# %%
## "Out of sample" predictions

df_predictions = caml.predict(
    out_of_sample_df=df,
    out_of_sample_uuid="uuid",
    return_predictions=False,
    join_predictions=True,
)

if df_backend == "pyspark":
    df_predictions.show()
else:
    print(df_predictions)

# %%
## Append to internal dataframe

caml.predict(
    out_of_sample_df=None,
    out_of_sample_uuid=None,
    join_predictions=True,
    return_predictions=False,
)

caml.dataframe

# %% [markdown]
# #### CATE Rank Ordering

# %%
## "Out of sample" predictions

df_rank_ordered = caml.rank_order(
    out_of_sample_df=df_predictions,
    return_rank_order=False,
    join_rank_order=True,
    treatment_category=1,
)

df_rank_ordered

# %%
## Append to internal dataframe

caml.rank_order(
    out_of_sample_df=None,
    return_rank_order=False,
    join_rank_order=True,
    treatment_category=1,
)

caml.dataframe

# %% [markdown]
# #### CATE Visualization/Summary

# %%
cate_summary = caml.summarize(out_of_sample_df=df_rank_ordered, treatment_category=1)

cate_summary

# %%
cate_summary = caml.summarize(out_of_sample_df=None, treatment_category=1)

cate_summary

# %%
true_ate

# %% [markdown]
# #### Access my dataframe and estimator object

# %%
caml.dataframe

# %%
from econml.score import EnsembleCateEstimator

# Use this estimator object as pickled object for optimized inference
final_estimator = caml.final_estimator

if isinstance(final_estimator, EnsembleCateEstimator):
    for model in final_estimator._cate_models:
        print(model)
        print(model._input_names)
else:
    print(final_estimator)
    print(final_estimator._input_names)
