import numpy as np
from econml.dml import LinearDML
from econml.score import RScorer


# Monkey patching Rscorer (Fixed in EconML PR - https://github.com/py-why/EconML/pull/927)
def patched_fit(
    self, y, T, X=None, W=None, sample_weight=None, groups=None, discrete_outcome=False
):
    if X is None:
        raise ValueError("X cannot be None for the RScorer!")

    self.lineardml_ = LinearDML(
        model_y=self.model_y,
        model_t=self.model_t,
        cv=self.cv,
        discrete_treatment=self.discrete_treatment,
        discrete_outcome=discrete_outcome,
        categories=self.categories,
        random_state=self.random_state,
        mc_iters=self.mc_iters,
        mc_agg=self.mc_agg,
    )
    self.lineardml_.fit(
        y,
        T,
        X=None,
        W=np.hstack([v for v in [X, W] if v is not None]),
        sample_weight=sample_weight,
        groups=groups,
        cache_values=True,
    )
    self.base_score_ = self.lineardml_.score_
    self.dx_ = X.shape[1]
    return self


RScorer.fit = patched_fit
