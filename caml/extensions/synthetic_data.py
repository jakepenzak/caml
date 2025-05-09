from typing import Callable, Collection

import numpy as np
import pandas as pd
import patsy
from doubleml.datasets import (
    make_heterogeneous_data,
    make_plr_CCDDHNR2018,
    make_plr_turrell2018,
)
from numpy.typing import ArrayLike
from scipy.linalg import toeplitz
from scipy.special import expit as sigmoid
from scipy.special import softmax
from typeguard import typechecked

from ..generics import experimental


def _truncate_and_renormalize_probabilities(
    prob_matrix: np.ndarray, epsilon: float = 0.01
) -> np.ndarray:
    """Truncate and renormalize probabilities (in case of softmax).

    Helps to ensure we satisify positivity/overlap in treatment probabilites and don't have probabilities too extreme.

    Parameters
    ----------
    prob_matrix : np.ndarray
        Matrix of probabilities to be truncated and renormalized of shape (n_samples, n_features)
    epsilon : float
        Value to clip probabilities on both ends of the distribution. Default is 0.05.

    Returns
    -------
    np.ndarray
        Matrix of probabilities

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.extensions.synthetic_data import _truncate_and_renormalize_probabilities

    probs = np.array([0.99,0.95,0.9,0.5,0.1,0.05,0.01])
    _truncate_and_renormalize_probabilities(probs, epsilon=0.05)
    ```
    """
    prob_matrix = np.clip(prob_matrix, epsilon, 1 - epsilon)
    if prob_matrix.ndim > 1:
        prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)

    return prob_matrix


@experimental
@typechecked
class SyntheticDataGenerator:
    r"""Generate highly flexible synthetic data for use in causal inference and CaML testing.

    **SyntheticDataGenerator is experimental and may change significantly in future versions.**

    The general form of the data generating process is:

    $$
    \mathbf{Y_i} = \tau (\mathbf{X_i}) \mathbf{T_i} + g(\mathbf{W_i}, \mathbf{X_i}) + \mathbf{\epsilon_i}
    $$
    $$
    \mathbf{T}_i=f(\mathbf{W}_i, \mathbf{X_{i,\mathcal{S}}})+\mathbf{\eta_i}
    $$

    where $\mathbf{Y_i}$ are the outcome(s), $\mathbf{T_i}$ are the treatment(s), $\mathbf{X_i}$ are the effect modifiers (leveraged for treatment effect heterogeneity)
    with an optional random subset $\mathcal{S}$ selected as confounders, $\mathbf{W_i}$ are the confounders, $\mathbf{\epsilon_i}$ and $\mathbf{\eta_i}$ are the error terms drawn from
    normal distributions with optional specified standard deviation, $\tau$ is the CATE function, $g$ is the linearly seperable/nuisance component of the outcome function,
    and $f$ is the treatment function. Note in the case of no modifier variables, we obtain a purely partially linear model, with $\tau$ as a constant.

    For linear data generating process, $f$ and $g$ consist of strictly linear terms and untransformed variables. $\tau(\mathbf{X_i})$ consists of linear interaction terms.

    For nonlinear data generating process, $f$ and $g$ are generated via Generalized Additive Models (GAMs) with randomly selected nonlinear transformations.
    $\tau(\mathbf{X_i})$ contains interaction terms with $\mathbf{X}$ and nonlinear transformations of $\mathbf{X}$.

    Note in the case of binary/discrete outcomes or treatments, sigmoid and softmax functions are used to transform log odds to probabilities.

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        X((X))-->Y((Y));
        W((W))-->Y((Y));
        W((W))-->T((T));
        X((X))-->|"S"|T((T));
        T((T))-->|"τ(X)"|Y((Y));

        linkStyle 0,1,2,3,4 stroke:black,stroke-width:2px
    ```
    </div>

    For a more detailed working example, see [SyntheticDataGenerator Example](../03_Examples/SyntheticDataGenerator.qmd).

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_cont_outcomes : int
        Number of continuous outcomes ($Y$).
    n_binary_outcomes : int
        Number of binary outcomes ($Y$).
    n_cont_treatments : int
        Number of continuous treatments ($T$).
    n_binary_treatments : int
        Number of binary treatments ($T$).
    n_discrete_treatments : int
        Number of discrete treatments ($T$).
    n_cont_confounders : int
        Number of continuous confounders ($W$).
    n_binary_confounders : int
        Number of binary confounders ($W$).
    n_discrete_confounders : int
        Number of discrete confounders ($W$).
    n_cont_modifiers : int
        Number of continuous treatment effect modifiers ($X$).
    n_binary_modifiers : int
        Number of binary treatment effect modifiers ($X$).
    n_discrete_modifiers : int
        Number of discrete treatment effect modifiers ($X$).
    n_confounding_modifiers : int
        Number of confounding treatment effect modifiers ($X_{\mathcal{S}}$).
    stddev_outcome_noise : float
        Standard deviation of the outcome noise ($\epsilon$).
    stddev_treatment_noise : float
        Standard deviation of the treatment noise ($\eta$).
    causal_model_functional_form : str
        Functional form of the causal model, can be "linear" or "nonlinear".
    n_nonlinear_transformations : int | None
        Number of nonlinear transformations, only applies if causal_model_functional_form="nonlinear".
    seed : int | None
        Random seed to use for generating the data.

    Attributes
    ----------
    df : pd.DataFrame
        The data generated by the data generation process.
    cates : pd.DataFrame
        The true conditional average treatment effects (CATEs) of the data.
    ates : pd.DataFrame
        The true average treatment effects (ATEs) of the data.
    dgp : dict
        The true data generating processes of the treatments and outcomes.
        Contains the design matrix formula, parameters, noise, raw_scores, and function used to generate the data.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    data_generator = SyntheticDataGenerator(n_cont_outcomes=1,
                                            n_binary_treatments=1,
                                            n_cont_confounders=2,
                                            n_cont_modifiers=2,
                                            seed=10)
    data_generator.df
    ```

    ```{python}
    data_generator.cates
    ```

    ```{python}
    data_generator.ates
    ```

    ```{python}
    for t, df in data_generator.dgp.items():
        print(f"\nDGP for {t}:")
        print(df)
    ```
    """

    def __init__(
        self,
        n_obs: int = 10_000,
        n_cont_outcomes: int = 1,
        n_binary_outcomes: int = 0,
        n_cont_treatments: int = 0,
        n_binary_treatments: int = 1,
        n_discrete_treatments: int = 0,
        n_cont_confounders: int = 0,
        n_binary_confounders: int = 0,
        n_discrete_confounders: int = 0,
        n_cont_modifiers: int = 0,
        n_binary_modifiers: int = 0,
        n_discrete_modifiers: int = 0,
        n_confounding_modifiers: int = 0,
        stddev_outcome_noise: float = 1.0,
        stddev_treatment_noise: float = 1.0,
        causal_model_functional_form: str = "linear",
        n_nonlinear_transformations: int | None = None,
        seed: int | None = None,
    ):
        if causal_model_functional_form not in ["linear", "nonlinear"]:
            raise ValueError(
                f"Invalid functional form. Must be choice of {['linear', 'nonlinear']}"
            )
        if n_cont_outcomes + n_binary_outcomes == 0:
            raise ValueError("At least one outcome variable type must be specified.")
        if n_cont_treatments + n_binary_treatments + n_discrete_treatments == 0:
            raise ValueError("At least one treatment variable type must be specified.")
        if n_obs <= 0:
            raise ValueError("Number of observations must be greater than 0.")

        self._n_obs = n_obs
        self._n_cont_outcomes = n_cont_outcomes
        self._n_binary_outcomes = n_binary_outcomes
        self._n_cont_treatments = n_cont_treatments
        self._n_binary_treatments = n_binary_treatments
        self._n_discrete_treatments = n_discrete_treatments
        self._n_cont_confounders = n_cont_confounders
        self._n_binary_confounders = n_binary_confounders
        self._n_discrete_confounders = n_discrete_confounders
        self._n_cont_modifiers = n_cont_modifiers
        self._n_binary_modifiers = n_binary_modifiers
        self._n_discrete_modifiers = n_discrete_modifiers
        self._n_confounding_modifiers = n_confounding_modifiers
        self._stddev_outcome_noise = stddev_outcome_noise
        self._stddev_treatment_noise = stddev_treatment_noise
        self._causal_model_functional_form = causal_model_functional_form
        self._n_nonlinear_transformations = (
            n_nonlinear_transformations
            if n_nonlinear_transformations is not None
            and causal_model_functional_form == "nonlinear"
            else 10
            if causal_model_functional_form == "nonlinear"
            else None
        )
        self._seed = seed if seed is not None else np.random.randint(1, 1000)
        self._rng = np.random.default_rng(seed)

        self._generate_data()

    @staticmethod
    def create_design_matrix(
        df: pd.DataFrame, formula: str, return_type: str = "dataframe", **kwargs
    ) -> pd.DataFrame | np.ndarray:
        """Create a design matrix from a formula and data.

        This method can be used to reconstruct the design matrices used to generate the treatment and outcome
        variables. Furthermore, using `dgp` attribute, using the returned design matrix, one can generate the original
        outcomes and treatment variables. See below example.

        Parameters
        ----------
        df : pd.DataFrame
            The input data.
        formula : str
            The formula to be used with patsy.
        return_type : str, optional
            The type of the returned design matrix. Can be either "dataframe" or "matrix". Default is "dataframe".
        **kwargs
            Additional keyword arguments to be passed to patsy.dmatrix.

        Returns
        -------
        pd.DataFrame | np.ndarray
            The design matrix.

        Examples
        --------
        ```{python}
        import numpy as np
        df = data_generator.df
        dgp = data_generator.dgp['Y1_continuous']

        design_matrix = data_generator.create_design_matrix(df,formula=dgp['formula'])

        print(design_matrix.columns)

        # Recreate Y1_continuous
        params = dgp['params']
        noise = dgp['noise']
        f = dgp['function']

        f(design_matrix,params,noise)

        assert np.allclose(f(design_matrix,params,noise), df['Y1_continuous'])
        ```
        """
        return patsy.dmatrix(formula, data=df, return_type=return_type, **kwargs)

    def _generate_data(self):
        """
        Execute the data generation process end-to-end.

        1. Generates randomly and independently drawn confounders from various distributions.
        2. Generates randomly and independently drawn heterogeneity inducing covariates (modifiers) from various distributions.
        3. Generates the treatment variables as a function of confounders and a random subset of heterogeneity inducing covariates, specified by `n_confounding_modifiers`. Returns true dgp.
        4. Generates the outcome variables as a function of confounders and treatment variables, with treatment interactions with heterogeneity inducing covariates (modifiers). Returns true dgp and treatment effects.
        5. Sets key attributes of the class including:
            - `df` - The simulated data
            - `cates` - The true conditional average treatment effects (CATEs)
            - `ates` - The true average treatment effects (ATEs)
            - `dgp` - The data generating process specs
        """
        # Generate confounders
        confounders = self._generate_independent_variables(
            n_continuous=self._n_cont_confounders,
            n_binary=self._n_binary_confounders,
            n_discrete=self._n_discrete_confounders,
            col_prefix="W",
        )

        # Generate modifiers
        modifiers = self._generate_independent_variables(
            n_continuous=self._n_cont_modifiers,
            n_binary=self._n_binary_modifiers,
            n_discrete=self._n_discrete_modifiers,
            col_prefix="X",
        )

        # Generate treatment variables
        subset_modifiers = modifiers.sample(
            n=self._n_confounding_modifiers, axis=1, random_state=self._seed
        )

        treatments, treatments_dgp, _ = self._generate_dependent_variables(
            dfs=[confounders, subset_modifiers],
            n_continuous=self._n_cont_treatments,
            n_binary=self._n_binary_treatments,
            n_discrete=self._n_discrete_treatments,
            stddev_err=self._stddev_treatment_noise,
            col_prefix="T",
        )

        # Generate outcome variables
        outcomes, outcomes_dgp, cates = self._generate_dependent_variables(
            dfs=[confounders, modifiers, treatments],
            n_continuous=self._n_cont_outcomes,
            n_binary=self._n_binary_outcomes,
            n_discrete=0,
            stddev_err=self._stddev_outcome_noise,
            col_prefix="Y",
            include_heterogeneity=True,
            include_treatment_effects=True,
        )

        # Combine variables into single dataframe
        synthetic_data = pd.concat(
            [confounders, modifiers, treatments, outcomes], axis=1
        )

        # Prettify CATEs and ATEs report
        cate_df, ate_df = self._treatment_effect_report(cates)

        self.df = synthetic_data
        self.cates = cate_df
        self.ates = ate_df
        self.dgp = {**treatments_dgp, **outcomes_dgp}

    def _generate_independent_variables(
        self,
        n_continuous: int,
        n_binary: int,
        n_discrete: int,
        col_prefix: str,
    ) -> pd.DataFrame:
        """Generate independently drawn variables from various distributions.

        This is used to generate confounder and mediator variables.

        TODO: Extend with covariance structure?

        Parameters
        ----------
        n_continuous : int
            Number of continuous variables to generate.
        n_binary : int
            Number of binary variables to generate.
        n_discrete : int
            Number of discrete variables to generate.
        col_prefix : str
            Prefix for column names.

        Returns
        -------
        pd.DataFrame
            Dataframe of independent variables.
        """
        ind_vars = {}

        var_types = (
            ["continuous"] * n_continuous
            + ["binary"] * n_binary
            + ["discrete"] * n_discrete
        )

        for i, t in enumerate(var_types):
            ind_vars[f"{col_prefix}{i + 1}_{t}"] = self._generate_random_variable(
                n_obs=self._n_obs, var_type=t, rng=self._rng
            )

        df = pd.DataFrame(ind_vars)
        if df.shape == (0, 0):
            df = pd.DataFrame(index=range(self._n_obs))
        return df

    @staticmethod
    def _generate_random_variable(
        n_obs: int,
        var_type: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates a random variable of n observations from randomly selected distribution given a data type.

        Parameters
        ----------
        n_obs : int
            Number of observations.
        var_type : str
            Type of the variable to generate, choose from "continuous", "binary" or "discrete".
        rng : np.random.Generator
            Numpy random number generator.

        Returns
        -------
        np.ndarray
            Generated random variable.

        """
        if var_type == "continuous":
            distributions = [
                "normal",
                "uniform",
                "exponential",
                "gamma",
                "beta",
                "laplace",
            ]

            dist = rng.choice(distributions)

            if dist == "normal":
                mean, std = rng.uniform(-5, 5), rng.uniform(0.5, 2)
                res = rng.normal(mean, std, n_obs)
            elif dist == "uniform":
                low, high = rng.uniform(-10, 0), rng.uniform(0, 10)
                res = rng.uniform(low, high, n_obs)
            elif dist == "exponential":
                scale = rng.uniform(1, 3)
                res = rng.exponential(scale, n_obs)
            elif dist == "gamma":
                shape, scale = rng.uniform(1, 3), rng.uniform(1, 3)
                res = rng.gamma(shape, scale, n_obs)
            elif dist == "beta":
                a, b = rng.uniform(1, 3), rng.uniform(1, 3)
                res = rng.beta(a, b, n_obs)
            else:  # Laplace
                loc, scale = rng.uniform(-5, 5), rng.uniform(0.5, 2)
                res = rng.laplace(loc, scale, n_obs)

        elif var_type == "binary":
            p = rng.uniform(0.1, 0.9)
            res = rng.binomial(1, p, n_obs)

        else:  # Discrete
            distributions = ["poisson", "geometric", "multinomial", "uniform"]

            dist = rng.choice(distributions)

            if dist == "poisson":
                lam = rng.uniform(1, 10)
                res = rng.poisson(lam, n_obs)

            elif dist == "geometric":
                p = rng.uniform(0.1, 0.9)
                res = rng.geometric(p, n_obs)

            elif dist == "multinomial":
                n_categories = rng.choice(range(2, 7))
                probs = rng.dirichlet(np.ones(n_categories))
                res = rng.choice(range(n_categories), size=n_obs, p=probs)

            else:  # Uniform
                n_categories = rng.choice(range(2, 7))
                res = rng.choice(range(0, n_categories), size=n_obs)

        return res

    def _generate_dependent_variables(
        self,
        dfs: Collection[pd.DataFrame],
        n_continuous: int,
        n_binary: int,
        n_discrete: int,
        col_prefix: str,
        stddev_err: float,
        include_heterogeneity: bool = False,
        include_treatment_effects: bool = False,
    ) -> tuple[pd.DataFrame, dict, dict | None]:
        """Generate treatment or outcome variables as functions of the confounders and/or mediators for synthetic data generation.

        This method returns:

        1. The generated dependent variable (outcome or treatment).
        2. The true data generating process, containing all components to replicate the data (design matrix formula, parameters, noise, and function).
        3. In the case of outcomes, the true individual-level effects/CATEs

        Parameters
        ----------
        dfs : Collection[pd.DataFrame]
            Dataframes to generate synthetic data from.
        n_continuous : int
            Number of continuous variables to generate.
        n_binary : int
            Number of binary variables to generate.
        n_discrete : int
            Number of discrete variables to generate.
        col_prefix : str
            Prefix for column names.
        stddev_err : float
            Standard deviation of error term.
        include_heterogeneity : bool, optional
            Whether to include heterogeneity in the data, by default False
        include_treatment_effects : bool, optional
            Whether to include treatment effects in the data, by default False

        Returns
        -------
        tuple[pd.DataFrame, dict, dict | None]
            Synthetic data, data generation parameters, and, optionally, true CATEs.
        """
        dependents = {}
        dgp = {}
        cates = {}
        combined_df = pd.concat(dfs, axis=1)

        dep_types = (
            ["continuous"] * n_continuous
            + ["binary"] * n_binary
            + ["discrete"] * n_discrete
        )

        for i, dep_type in enumerate(dep_types):
            col_name = f"{col_prefix}{i + 1}_{dep_type}"
            dependents[col_name], dgp[col_name], cates[col_name] = self._dgp(
                df=combined_df,
                stddev_err=stddev_err,
                dep_type=dep_type,
                n_nonlinear_transformations=self._n_nonlinear_transformations,
                include_heterogeneity=include_heterogeneity,
                include_treatment_effects=include_treatment_effects,
            )

        return pd.DataFrame(dependents), dgp, cates

    def _dgp(
        self,
        df: pd.DataFrame,
        stddev_err: float,
        dep_type: str,
        n_nonlinear_transformations: int | None = None,
        include_heterogeneity: bool = False,
        include_treatment_effects: bool = False,
    ):
        """Calls methods to create formula, design matrix, dgp, and treatments for each individual dependent variable.

        Helper function for `_generate_dependent_variables`.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the features.
        stddev_err : float
            The standard deviation of the error term.
        dep_type : str
            The type of dependent variable to generate ().
        n_nonlinear_transformations : int | None
            The number of nonlinear transformations to apply.
        include_heterogeneity : bool
            Whether to include heterogeneity in the dependent variable.
        include_treatment_effects : bool
            Whether to include treatment effects in the dependent variable.

        Returns
        -------
        pd.DataFrame
            The dataframe containing the dependent variables.
        dict
            The dictionary containing the design matrix, parameters, noise, raw scores, and function.
        dict | None
            The dictionary containing the treatment effects, or None if not included.

        """
        formula = self._create_patsy_formula(
            df=df,
            n_nonlinear_transformations=n_nonlinear_transformations,
            include_heterogeneity=include_heterogeneity,
            seed=self._seed,
        )

        if formula is not None:
            d_matrix = self.create_design_matrix(
                df,
                formula=formula,
            )
        else:
            d_matrix = df

        dep, params, noise, raw_scores, f = self._create_dgp_function(
            df=d_matrix,
            n_obs=self._n_obs,
            stddev_err=stddev_err,
            dep_type=dep_type,
            rng=self._rng,
        )
        if include_treatment_effects:
            cates = self._compute_treatment_effects(f, params, noise, df, formula)
        else:
            cates = None

        dgp = {
            "formula": formula,
            "params": params,
            "noise": noise,
            "raw_scores": raw_scores,
            "function": f,
        }
        return dep, dgp, cates

    @staticmethod
    def _create_patsy_formula(
        df: pd.DataFrame,
        n_nonlinear_transformations: int | None = None,
        include_heterogeneity: bool = False,
        seed: int | None = None,
    ) -> str | None:
        """Create design matrix formula to be used with patsy.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with features to be included in the design matrix.
        n_nonlinear_transformations : int | None, optional
            Number of nonlinear transformations to apply to the covariates, by default None
        include_heterogeneity : bool, optional
            Whether to include heterogeneity in the design matrix (via interaction terms), by default False
        seed : int | None, optional
            Seed for the random number generator, by default None

        Returns
        -------
        str | None
            Design matrix formula to be used with patsy.
        """
        columns = df.columns
        if len(columns) == 0:
            return

        formula = "1 + " + " + ".join(columns)

        non_treat_columns = [c for c in columns if "T" not in c]
        if n_nonlinear_transformations is not None and len(non_treat_columns) > 0:
            np.random.seed(seed)

            transformations = [
                # lambda x: f"I({x}**2)",  # square(x)
                lambda x: f"np.log(np.abs({x})+0.01)",  # log(abs(x)+0.01)
                lambda x: f"np.sqrt(np.abs({x}))",  # sqrt(abs(x))
                lambda x: f"np.sin({x})",  # sine(x)
                lambda x: f"np.cos({x})",  # cosine(x)
            ]

            interaction_prob = 1 / (1 + len(transformations))
            terms = set()
            while len(terms) < n_nonlinear_transformations:
                if np.random.uniform() < interaction_prob and len(df) > 2:
                    x1, x2 = np.random.choice(non_treat_columns, 2)
                    term = f"{x1}*{x2}"
                else:
                    x = np.random.choice(non_treat_columns)
                    transform = np.random.choice(transformations)
                    term = transform(x)
                terms.add(term)

            formula = formula + " + " + " + ".join(sorted(list(terms)))

        if include_heterogeneity:
            treat_columns = [c for c in columns if "T" in c]
            interactions = set()
            for treat_column in treat_columns:
                for term in formula.split(" + "):
                    if "T" not in term and "W" not in term and term != "1":
                        interactions.add(f"{treat_column}*{term}")
            if len(interactions) > 0:
                formula = formula + " + " + " + ".join(sorted(list(interactions)))

        return formula

    @staticmethod
    def _create_dgp_function(
        df: pd.DataFrame,
        n_obs: int,
        stddev_err: float,
        dep_type: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable]:
        """Creates the data generation process (DGP) function & simulates the data.

        Parameters
        ----------
        df : pd.DataFrame
            The input data.
        n_obs : int
            The number of observations.
        stddev_err : float
            The standard deviation of the error term.
        dep_type : str
            The type of the dependent variable.
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable]
            The generated dependent variable, params, noise, raw scores, and the DGP function.
        """
        n_feats = df.shape[1]

        if dep_type == "discrete":
            n_categories = rng.choice(range(3, 6))
            param_size = (n_feats, n_categories)
            noise_size = (n_obs, n_categories)
        else:
            n_categories = 1
            param_size = n_feats
            noise_size = n_obs

        if dep_type != "continuous":
            params = rng.normal(0, 0.5, size=param_size)
        else:
            params = rng.normal(0, 2, size=param_size)
        noise = rng.normal(0, stddev_err, size=noise_size)

        if dep_type == "continuous":

            @typechecked
            def f_cont(x: pd.DataFrame, params: ArrayLike, noise: ArrayLike):
                """Continuous target function."""
                return x @ params + noise

            f = f_cont
            scores = np.array(f(df, params, noise))
            dep = scores
        elif dep_type == "binary":

            @typechecked
            def f_binary(x: pd.DataFrame, params: ArrayLike, noise: ArrayLike):
                """Binary target function."""
                raw = x @ params + noise

                probs = sigmoid(raw)
                return _truncate_and_renormalize_probabilities(probs)

            f = f_binary
            scores = np.array(f(df, params, noise))
            dep = rng.binomial(1, scores)
        else:  # Discrete

            @typechecked
            def f_discrete(x: pd.DataFrame, params: ArrayLike, noise: ArrayLike):
                """Discrete target function."""
                raw = x @ params + noise

                probs = softmax(raw)
                return _truncate_and_renormalize_probabilities(probs)

            f = f_discrete
            scores = np.array(f(df, params, noise))
            dep = np.array([rng.choice(range(n_categories)) for prob in scores])

        return dep, params, noise, scores, f

    def _compute_treatment_effects(
        self,
        f: Callable,
        params: np.ndarray,
        noise: np.ndarray,
        df: pd.DataFrame,
        formula: str,
    ) -> dict:
        """
        Function to call & compute true treatment effects.

        Parameters
        ----------
        f : Callable
            Outcome function.
        params : np.ndarray
            Parameters for dgp to be used in f.
        noise : np.ndarray
            Exogenous error term to be used in f.
        df : pd.DataFrame
            The dataframe including raw variables.
        formula : str
            Formula used to generate design matrix.

        Returns
        -------
        dict
            Dictionary of true treatment effects.

        """
        cates = {}
        for t in [c for c in df.columns if c.count("_") == 1 and "T" in c]:
            if "continuous" in t:
                levels = ["continuous"]
            elif "binary" in t:
                levels = [0, 1]
            else:  # Discrete
                levels = df[t].unique().tolist()

            cates[t] = self._compute_potential_outcome_differences(
                f=f,
                params=params,
                noise=noise,
                df=df,
                wrt=t,
                formula=formula,
                levels=levels,
            )

        return cates

    def _compute_potential_outcome_differences(
        self,
        f: Callable,
        params: np.ndarray,
        noise: np.ndarray,
        df: pd.DataFrame,
        wrt: str,
        formula: str,
        levels: list,
    ) -> dict[str, ArrayLike] | ArrayLike:
        """Computes potential outcome differences of some outcome function for each individual, returning the conditional average treatment effects (CATEs).

        Parameters
        ----------
        f : Callable
            Outcome function.
        params : np.ndarray
            Parameters for dgp to be used in f.
        noise : np.ndarray
            Exogenous error term to be used in f.
        df : pd.DataFrame
            Dataframe containing covariates and treatment.
        wrt : str
            With respect to, the name of treatment variable.
        formula : str
            Formula used to create design matrix.
        levels : list
            The treatment leves to capture potential outcome differences for. For continuous treatments, use 'continuous', which measures one unit change in treatment.

        Returns
        -------
        dict[str, np.ndarray] | np.ndarray
            The conditional average treatment effects (CATEs). If levels is a list with more than [0,1], returns a dictionary of CATEs for each level.

        """
        cates = {}
        for lev in levels:
            if lev == 0:
                continue
            else:
                data_treat = df.copy()
                data_control = df.copy()

                if lev == "continuous":
                    data_treat[wrt] = data_treat[wrt] + 1
                else:
                    data_treat[wrt] = lev
                    data_control[wrt] = 0

                design_treat = self.create_design_matrix(data_treat, formula=formula)
                design_control = self.create_design_matrix(
                    data_control, formula=formula
                )
                noise = np.zeros_like(noise)
                cates[f"{lev}_v_0"] = f(design_treat, params, noise) - f(
                    design_control, params, noise
                )

        if len(cates) == 1:
            return list(cates.values())[0]
        else:
            return cates

    @staticmethod
    def _treatment_effect_report(
        cates: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a prettified dataframe of the true conditional average treatment effects (CATEs) and average treatment effects (ATE).

        Parameters
        ----------
        cates : dict[str, dict]
            A dictionary including key as outcome name and value as dictionary of CATES of each treatment on that outcome.

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            Prettified dataframe of the CATEs and ATEs.
        """
        dict_effects = {}
        for outcome, effects in cates.items():
            for treatment, values in effects.items():
                var = f"CATE_of_{treatment}_on_{outcome}"
                if isinstance(values, dict):
                    for levels, results in values.items():
                        var_lev = var + f"_level_{levels}"
                        dict_effects[var_lev] = results
                else:
                    dict_effects[var] = values

        cate_df = pd.DataFrame(dict_effects)

        ate_df = cate_df.mean(axis=0).reset_index()
        ate_df.columns = ["Treatment", "ATE"]
        ate_df["Treatment"] = ate_df["Treatment"].str.replace("CATE_of_", "")

        return cate_df, ate_df


@typechecked
def make_partially_linear_dataset_simple(
    n_obs: int = 1000,
    n_confounders: int = 5,
    dim_heterogeneity: int = 2,
    binary_treatment: bool = True,
    seed: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    r"""Simulate data generating process from a partially linear model with a simple 1 or 2 dimensional CATE function.

    The outcome is continuous and the treatment can be binary or continuous. The dataset is generated using the `make_heterogeneous_data` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is, in the case of dim_heterogeneity=1:

    $$
    y_i= \tau (x_0) d_i + g(\mathbf{X_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{X_i})+\eta_i
    $$

    or, in the case of dim_heterogeneity=2:

    $$
    y_i= \tau (x_0,x_1) d_i + g(\mathbf{X_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{X_i})+\eta_i
    $$

    where $y_i$ is the outcome, $d_i$ is the treatment, $\mathbf{X_i}$ are the confounders, $\epsilon_i$ and $\eta_i$ are the error terms, $\\tau$ is the CATE function, $g$ is the outcome function, and $f$ is the treatment function.

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    Here the ATE is defined as the average of the CATE function over all observations: $\mathbb{E}[\tau (\cdot)]$

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        Xn((X))-->d((d));
        Xn((X))-->y((y));
        d((d))-->|"τ(x0,x1)"|y((y));

        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    n_confounders : int
        The number of confounders $X$.
    dim_heterogeneity : int
        The dimension of the heterogeneity $x_0$ or $(x_0,x_1)$. Can only be 1 or 2.
    binary_treatment : bool
        Whether the treatment $d$ is binary or continuous.
    seed : int | None
        The seed to use for the random number generator.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and X are the confounders with a 1d or 2d subset utilized for heterogeneity.
    true_cates : numpy.ndarray
        The true conditional average treatment effects.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import make_partially_linear_dataset_simple
    df, true_cates, true_ate = make_partially_linear_dataset_simple(n_obs=1000,
                                                                    n_confounders=5,
                                                                    dim_heterogeneity=2,
                                                                    binary_treatment=True,
                                                                    seed=1)

    print(f"True CATES: {true_cates[:5]}")
    print(f"True ATE: {true_ate}")
    print(df.head())
    ```
    """
    if dim_heterogeneity not in [1, 2]:
        raise ValueError("dim_heterogeneity must be 1 or 2.")

    np.random.seed(seed)

    data = make_heterogeneous_data(
        n_obs=n_obs,
        p=n_confounders,
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
) -> tuple[pd.DataFrame, np.ndarray, float]:
    r"""Simulate a data generating process from a partially linear model with a constant treatment effect (ATE only).

    The outcome and treatment are both continuous.The dataset is generated using the `make_plr_CCDDHNR2018` or `make_plr_turrell2018` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is:

    $$
    y_i= \tau_0 d_i + g(\mathbf{W_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{W_i})+\eta_i
    $$

    where $y_i$ is the outcome, $d_i$ is the treatment, $\mathbf{W_i}$ are the confounders, $\epsilon_i$ and $\eta_i$ are the error terms, $\tau_0$ is the ATE parameter, $g$ is the outcome function, and $f$ is the treatment function.

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        W((W))-->d((d));
        W((W))-->y((y));
        d((d))-->|"τ0"|y((y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    ate : float
        The average treatment effect $\tau_0$.
    n_confounders : int
        The number of confounders $\mathbf{W_i}$ to generate.
    dgp : str
        The data generating process to use. Can be "make_plr_CCDDHNR20" or "make_plr_turrell2018".
    seed : int | None
        The seed to use for the random number generator.
    **doubleml_kwargs
        Additional keyword arguments to pass to the data generating process.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and W are the confounders.
    true_cates : numpy.ndarray
        The true conditional average treatment effects, which are all equal to the ATE here.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import make_partially_linear_dataset_constant
    df, true_cates, true_ate = make_partially_linear_dataset_constant(n_obs=1000,
                                                        ate=4.0,
                                                        n_confounders=10,
                                                        dgp="make_plr_CCDDHNR2018",
                                                        seed=1)

    print(f"True CATES: {true_cates[:5]}")
    print(f"True ATE: {true_ate}")
    print(df.head())
    ```
    """
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

    df.columns = [c.replace("X", "W") for c in df.columns if "X" in c] + ["y", "d"]

    true_ate = ate
    true_cates = np.full(n_obs, true_ate)

    return df, true_cates, true_ate


@typechecked
def make_fully_heterogeneous_dataset(
    n_obs: int = 1000,
    n_confounders: int = 5,
    theta: float = 4.0,
    seed: int | None = None,
    **doubleml_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    r"""Simulate data generating process from an interactive regression model with fully heterogenous treatment effects.

    The outcome is continuous and the treatment is binary. The dataset is generated using a modified version of `make_irm_data` function from the [`doubleml` package](https://docs.doubleml.org/stable/index.html).

    The general form of the data generating process is:

    $$
    y_i= g(d_i,\mathbf{X_i})+\epsilon_i
    $$
    $$
    d_i=f(\mathbf{X_i})+\eta_i
    $$

    where $y_i$ is the outcome, $d_i$ is the treatment, $\mathbf{X_i}$ are the confounders utilized for full effect heterogeneity, $\epsilon_i$ and $\eta_i$ are the error terms, $g$ is the outcome function, and $f$ is the treatment function.

    See the `doubleml` documentation for more details on the specific functional forms of the data generating process.

    Note that the treatment effect is fully heterogenous, thus the CATE is defined as: $\tau = \mathbb{E}[g(1,\mathbf{X}) - g(0,\mathbf{X})|\mathbf{X}]$ for any $\mathbf{X}$.

    The ATE is defined as the average of the CATE function over all observations: $\mathbb{E}[\tau (\cdot)]$

    As a DAG, the data generating process can be roughly represented as:

    <div style="text-align: center;">
    ```{mermaid}
    flowchart TD;
        X((X))-->d((d));
        X((X))-->y((y));
        d((d))-->|"τ(X)"|y((y));
        linkStyle 0,1 stroke:black,stroke-width:2px
        linkStyle 1,2 stroke:black,stroke-width:2px
    ```
    </div>

    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    n_confounders : int
        The number of confounders $\mathbf{X_i}$ to generate (these are utilized fully for heterogeneity).
    theta : float
        The base parameter for the treatment effect. Note this can differ slightly from the true ATE.
    seed : int | None
        The seed to use for the random number generator.
    **doubleml_kwargs
        Additional keyword arguments to pass to the data generating process.

    Returns
    -------
    df : pandas.DataFrame
        The generated dataset where y is the outcome, d is the treatment, and X are the confounders which are fully utilized for heterogeneity.
    true_cates : numpy.ndarray
        The true conditional average treatment effects.
    true_ate : float
        The true average treatment effect.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import make_fully_heterogeneous_dataset
    df, true_cates, true_ate = make_fully_heterogeneous_dataset(n_obs=1000,
                                                                n_confounders=5,
                                                                theta=4.0,
                                                                seed=1)

    print(f"True CATEs: {true_cates[:5]}")
    print(f"True ATE: {true_ate}")
    print(df.head())
    ```
    """
    np.random.seed(seed)

    v = np.random.uniform(
        size=[
            n_obs,
        ]
    )
    zeta = np.random.standard_normal(
        size=[
            n_obs,
        ]
    )

    cov_mat = toeplitz([np.power(0.5, k) for k in range(n_confounders)])
    x = np.random.multivariate_normal(
        np.zeros(n_confounders),
        cov_mat,
        size=[
            n_obs,
        ],
    )

    R2_y = doubleml_kwargs.get("R2_y", 0.5)
    R2_d = doubleml_kwargs.get("R2_d", 0.5)

    beta = [1 / (k**2) for k in range(1, n_confounders + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_y = np.sqrt(R2_y / ((1 - R2_y) * b_sigma_b))
    c_d = np.sqrt(np.pi**2 / 3.0 * R2_d / ((1 - R2_d) * b_sigma_b))

    xx = np.exp(np.dot(x, np.multiply(beta, c_d)))
    d = 1.0 * ((xx / (1 + xx)) > v)

    def y_func(d, x, theta):
        return d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta

    y = y_func(d, x, theta)

    x_cols = [f"X{i + 1}" for i in np.arange(n_confounders)]
    df = pd.DataFrame(np.column_stack((x, y, d)), columns=x_cols + ["y", "d"])

    d1 = np.ones_like(d)
    d0 = np.zeros_like(d)

    true_cates = y_func(d1, x, theta) - y_func(d0, x, theta)
    true_ate = true_cates.mean()

    return df, true_cates, true_ate
