import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

plt.style.use("ggplot")


def cate_histogram_plot(
    estimated_cates: ArrayLike,
    *,
    true_cates: ArrayLike | None = None,
    figure_kwargs: dict = {},
    hist_kwargs: dict = {},
) -> plt.Figure:
    """
    Plots a histogram the estimated CATEs.

    Parameters
    ----------
    estimated_cates : numpy.typing.ArrayLike
        The estimated CATEs.

    true_cates : numpy.typing.ArrayLike | None
        The true CATEs.

    figure_kwargs : dict
        Matplotlib figure arguments.

    hist_kwargs : dict
        Matplotlib hist arguments.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The histogram figure object.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.extensions.plots import cate_histogram_plot

    np.random.seed(42)
    true_cates = np.random.normal(0, 1, 1000)
    estimated_cates = true_cates + np.random.normal(0, 0.5, 1000)

    fig = cate_histogram_plot(estimated_cates, true_cates=true_cates, hist_kwargs={'bins': 25})
    fig
    ```
    """

    _hist_kwargs = {
        "bins": 50,
        "color": "green",
        "alpha": 0.7,
        "label": "Estimated CATEs",
    }
    if hist_kwargs is not None:
        _hist_kwargs.update(hist_kwargs)

    fig, ax = plt.subplots(**figure_kwargs)
    if true_cates is not None:
        _hist_kwargs_true = _hist_kwargs.copy()
        _hist_kwargs_true.update({"color": "red", "label": "True CATEs", "alpha": 0.4})
        ax.hist(true_cates, **_hist_kwargs_true)
    ax.hist(estimated_cates, **_hist_kwargs)
    ax.set_xlabel("CATEs", fontsize=14)
    ax.legend()

    plt.close(fig)

    return fig


def cate_true_vs_estimated_plot(
    true_cates: ArrayLike,
    estimated_cates: ArrayLike,
    *,
    figure_kwargs: dict = {},
    scatter_kwargs: dict = {},
) -> plt.Figure:
    """
    Plots a scatter plot of the estimated CATEs against the true CATEs.

    Parameters
    ----------
    estimated_cates : numpy.typing.ArrayLike
        The estimated CATEs.

    figure_kwargs : dict
        Matplotlib figure arguments.

    scatter_kwargs : dict
        Matplotlib line arguments.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The line plot figure object.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.extensions.plots import cate_true_vs_estimated_plot

    np.random.seed(42)
    true_cates = np.random.normal(0, 1, 100)
    estimated_cates = true_cates + np.random.normal(0, 0.5, 100)

    fig = cate_true_vs_estimated_plot(true_cates, estimated_cates)
    fig
    ```
    """

    _scatter_kwargs = {
        "color": "green",
        "alpha": 0.7,
        "s": 10,
    }
    if scatter_kwargs is not None:
        _scatter_kwargs.update(scatter_kwargs)

    fig, ax = plt.subplots(**figure_kwargs)
    ax.scatter(true_cates, estimated_cates, label="Estimated CATEs", **_scatter_kwargs)

    ax.plot(
        np.sort(true_cates),
        np.sort(true_cates),
        alpha=0.7,
        label="Perfect model",
        color="black",
    )
    ax.set_xlabel("True CATEs", fontsize=14)
    ax.set_ylabel("Estimated CATEs", fontsize=14)
    ax.legend()

    plt.close(fig)

    return fig


def cate_line_plot(
    estimated_cates: ArrayLike,
    *,
    true_cates: ArrayLike | None = None,
    standard_errors: ArrayLike | None = None,
    alpha: float = 0.05,
    window: int = 30,
    figure_kwargs: dict = {},
    line_kwargs: dict = {},
) -> plt.Figure:
    """
    Plots a line plot of the ordered estimated CATEs as a rolling mean with optional confidence intervals.

    Parameters
    ----------
    estimated_cates : numpy.typing.ArrayLike
        The estimated CATEs.

    true_cates : numpy.typing.ArrayLike | None
        The true CATEs.

    standard_errors : numpy.typing.ArrayLike | None
        The standard errors of the estimated CATEs.

    alpha : float
        The alpha level for the confidence intervals. The default is 0.05, which corresponds to 95% confidence intervals.

    window : int
        The window size for the moving average.

    figure_kwargs : dict
        Matplotlib figure arguments.

    line_kwargs : dict
        Matplotlib line arguments.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The line plot figure object.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.extensions.plots import cate_line_plot

    np.random.seed(42)
    true_cates = np.random.normal(0, 1, 100)
    estimated_cates = true_cates + np.random.normal(0, 0.5, 100)
    standard_errors = np.abs(np.random.normal(0, 0.1, 100))

    fig = cate_line_plot(estimated_cates, true_cates=true_cates, standard_errors=standard_errors, window=5)
    fig
    ```
    """

    _line_kwargs = {
        "color": "green",
        "alpha": 0.7,
        "label": "Estimated CATEs",
    }
    if line_kwargs is not None:
        _line_kwargs.update(line_kwargs)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    fig, ax = plt.subplots(**figure_kwargs)
    if true_cates is not None:
        true_cate_ma = moving_average(np.sort(true_cates), window)
        _line_kwargs_true = _line_kwargs.copy()
        _line_kwargs_true.update({"color": "red", "label": "True CATEs", "alpha": 0.4})
        ax.plot(true_cate_ma, label="True CATEs")

    if standard_errors is not None:
        cate_w_se = np.hstack(
            [estimated_cates.reshape(-1, 1), standard_errors.reshape(-1, 1)]
        )
        cate_w_se_sorted = cate_w_se[np.argsort(cate_w_se[:, 0])]
        cate_ma = moving_average(cate_w_se_sorted[:, 0], window)
        se_ma = moving_average(cate_w_se_sorted[:, 1], window)
        confidence = 1 - alpha
        cv = norm.ppf(1 - alpha / 2)
        ax.fill_between(
            np.arange(len(estimated_cates) - window + 1),
            cate_ma - cv * se_ma,
            cate_ma + cv * se_ma,
            color="green",
            alpha=0.2,
            label=f"{confidence*100:.0f}% CI",
        )
    else:
        cate_ma = moving_average(np.sort(estimated_cates), window)

    ax.plot(cate_ma, **_line_kwargs)
    ax.set_xlabel("Number Observations", fontsize=14)
    ax.set_ylabel("CATEs", fontsize=14)
    ax.legend()

    plt.close(fig)

    return fig
