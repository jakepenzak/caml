import numpy as np
import pandas as pd
from doubleml.datasets import (
    make_confounded_irm_data,
    make_heterogeneous_data,
    make_plr_CCDDHNR2018,
    make_plr_turrell2018,
)
from typeguard import typechecked


@typechecked
def make_partially_linear_dataset_simple(
    n_obs: int = 1000,
    n_covariates: int = 10,
    n_confounders: int = 5,
    dim_heterogeneity: int = 2,
    binary_treatment: bool = True,
    seed: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Generate a partially linear data generating process with simple 1 or 2 dimensional CATE function. The outcome is continuous and the treatment can be binary or continuous.
    The dataset is generated using the`make_heterogeneous_data` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is, in the case of dim_heterogeneity=1:

    $$
    Y_i= \\tau (x_0) D_i + g(\mathbf{X_i},\mathbf{W_i})+\epsilon_i
    $$
    $$
    D_i=f(\mathbf{X_i})+\eta_i
    $$

    or, in the case of dim_heterogeneity=2:

    $$
    Y_i= \\tau (x_0,x_1) D_i + g(\mathbf{X_i},\mathbf{W_i})+\epsilon_i
    $$
    $$
    D_i=f(\mathbf{X_i})+\eta_i
    $$

    where $Y_i$ is the outcome, $D_i$ is the treatment, $\mathbf{X_i}$ are the confounders, $\mathbf{W_i}$ are the covariates that covary with $Y_i$ only, $\epsilon_i$ and $\eta_i$ are the error terms, $\\tau$ is the CATE function, $g$ is the outcome function, and $f$ is the treatment function.

    Here the ATE is defined as the average of the CATE function over the covariates: $\mathbb{E}[\\tau (\cdot)]$

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    As a DAG, the data generating process can be roughly represented as, where Xn is all confounders less x0 and x1:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        X0((x0))-->D((D));
        X0((x0))-->Y((Y));
        X1((x1))-->D((D));
        X1((x1))-->Y((Y));
        Xn((Xn))-->D((D));
        Xn((Xn))-->Y((Y));
        W((W))-->Y((Y));
        D((D))-->|"τ(x0,x1)"|Y((Y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Note, the returned data will not differentiate between X and W. A random subset of W is selected as confounders based on n_confounders.

    Parameters
    ----------
    n_obs : int
        The number of observations to generate. Default is 1000.
    n_covariates : int
        The number of covariates to generate. Default is 10.
    n_confounders : int
        The number of covariates that are confounders. Default is 5.
    dim_heterogeneity : int
        The dimension of the heterogeneity. Default is 2. Can only be 1 or 2.
    binary_treatment : bool
        Whether the treatment is binary or continuous. Default is True.
    seed : int | None
        The seed to use for the random number generator. Default is None.

    Returns
    -------
    df : pd.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and X are the covariates.
    true_cates : np.ndarray
        The true conditional average treatment effects.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    >>> from caml.extensions.synthetic_data import make_partially_linear_dataset_simple
    >>> df, true_cates, true_ate = make_partially_linear_simple_dataset(n_obs=1000, n_covariates=10, n_confounders=5, dim_heterogeneity=2, binary_treatment=True, seed=1)
    """

    if dim_heterogeneity not in [1, 2]:
        raise ValueError("dim_heterogeneity must be 1 or 2.")

    if seed is not None:
        np.random.seed(seed)

    data = make_heterogeneous_data(
        n_obs=n_obs,
        p=n_covariates,
        support_size=n_confounders,
        n_x=dim_heterogeneity,
        binary_treatment=binary_treatment,
    )

    df = pd.DataFrame(data["data"])
    df.columns = [c.replace("X_", "X") for c in df.columns]
    true_cates = data["effects"]
    true_ate = true_cates.mean()
    return df, true_cates, true_ate


@typechecked
def make_partially_linear_dataset_constant(
    n_obs: int = 1000,
    ate: float = 4.0,
    n_confounders: int = 10,
    dgp: str = "make_plr_CCDDHNR2018",
    seed: int | None = None,
    **doubleml_kwargs,
) -> tuple[pd.DataFrame, float]:
    """
    Generate a partially linear model data generating process with a constant treatment effect (ATE only). The outcome and treatment are both continuous.
    The dataset is generated using the `make_plr_CCDDHNR2018` or `make_plr_turrell2018` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is:

    $$
    Y_i= \\tau_0 D_i + g(\mathbf{X_i})+\epsilon_i
    $$
    $$
    D_i=f(\mathbf{X_i})+\eta_i
    $$

    where $Y_i$ is the outcome, $D_i$ is the treatment, $\mathbf{X_i}$ are the covariates, $\epsilon_i$ and $\eta_i$ are the error terms, $\\tau_0$ is the ATE parameter, $g$ is the outcome function, and $f$ is the treatment function.

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        X((X))-->D((D));
        X((X))-->Y((Y));
        D((D))-->|"τ0"|Y((Y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate. Default is 1000.
    ate : float
        The average treatment effect. Default is 4.0.
    n_confounders : int
        The number of confounders to generate. Default is 10.
    dgp : str
        The data generating process to use. Default is "make_plr_CCDDHNR20". Can be "make_plr_CCDDHNR20" or "make_plr_turrell2018".
    seed : int | None
        The seed to use for the random number generator. Default is None.
    **doubleml_kwargs
        Additional keyword arguments to pass to the data generating process.

    Returns
    -------
    df : pd.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and X are the covariates.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    >>> from caml.extensions.synthetic_data import make_partially_linear_dataset_constant
    >>> df, true_ate = make_partially_linear_dataset_constant(n_obs=1000, ate=4.0, n_confounders=10, dgp="make_plr_CCDDHNR2018", seed=1)
    """

    if seed is not None:
        np.random.seed(seed)

    if dgp == "make_plr_CCDDHNR2018":
        df = make_plr_CCDDHNR2018(
            n_obs=n_obs,
            dim_x=n_confounders,
            alpha=ate,
            return_type="DataFrame",
            **doubleml_kwargs,
        )
    elif dgp == "make_plr_turrell2018":
        df = make_plr_turrell2018(
            n_obs=n_obs,
            dim_x=n_confounders,
            theta=ate,
            return_type="DataFrame",
            **doubleml_kwargs,
        )
    else:
        raise ValueError(
            "dgp must be 'make_plr_CCDDHNR2018' or 'make_plr_turrell2018'."
        )

    true_ate = ate

    return df, true_ate


@typechecked
def make_fully_hetereogenous_dataset(
    n_obs: int = 1000,
    n_confounders: int = 5,
    ate: float = 4.0,
    seed: int | None = None,
    **doubleml_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Generate an "interactive regression" model data generating process with fully heterogenous treatment effects. The outcome is continuous and the treatment is binary.
    The dataset is generated using the `make_confounded_irm_data` function from the `doubleml` [package](https://docs.doubleml.org/stable/index.html).
    We enforce the additional "unobserved" confounder A to be zero for all observations, since confounding is captured in X.

    The general form of the data generating process is:

    $$
    Y_i= g(D_i,\mathbf{X_i})+\epsilon_i
    $$
    $$
    D_i=f(\mathbf{X_i})+\eta_i
    $$

    where $Y_i$ is the outcome, $D_i$ is the treatment, $\mathbf{X_i}$ are the covariates, $\epsilon_i$ and $\eta_i$ are the error terms, $g$ is the outcome function, and $f$ is the treatment function.

    Note that the treatment effect is fully heterogenous, thus the CATE is defined as: $\\tau = \\mathbb{E}[g(1,\mathbf{X}) - g(0,\mathbf{X})|\mathbf{X}]$ for any $\mathbf{X}$.

    The ATE is defined as the average of the CATE function over the covariates: $\mathbb{E}[\\tau (\cdot)]$

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        X((X))-->D((D));
        X((X))-->Y((Y));
        D((D))-->|"τ(X)"|Y((Y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate. Default is 1000.
    n_confounders : int
        The number of confounders to generate. Default is 5.
    ate : float
        The average treatment effect. Default is 4.0.
    seed : int | None
        The seed to use for the random number generator. Default is None.
    **doubleml_kwargs
        Additional keyword arguments to pass to the data generating process.

    Returns
    -------
    df : pd.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and X are the covariates.
    true_cates : pd.DataFrame
        The true conditional average treatment effects.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    >>> from caml.extensions.synthetic_data import make_fully_hetereogenous_dataset
    >>> df, true_cates, true_ate = make_fully_hetereogenous_dataset(n_obs=1000, n_confounders=5, ate=4.0, seed=1)
    """

    if seed is not None:
        np.random.seed(seed)

    data = make_confounded_irm_data(
        n_obs=n_obs,
        dim_x=n_confounders,
        theta=ate,
        return_type="DataFrame",
        gamma_a=0,
        beta_a=0,
        **doubleml_kwargs,
    )

    df = pd.DataFrame(data["x"], columns=[f"X{i}" for i in range(1, n_confounders + 1)])
    df["y"] = data["y"]
    df["d"] = data["d"]

    true_cates = data["oracle_values"]["y_1"] - data["oracle_values"]["y_0"]
    true_ate = true_cates.mean()

    return df, true_cates, true_ate
