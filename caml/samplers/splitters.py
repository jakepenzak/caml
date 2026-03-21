"""Cross-validation splitter utilities.

Provides thin helpers around `~~sklearn.model_selection.KFold`,
`~~sklearn.model_selection.GroupKFold`,
`~~sklearn.model_selection.StratifiedKFold`, and
`~~sklearn.model_selection.StratifiedGroupKFold`.
"""

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


def create_splitter(cv=3, groups=None, stratified=False, random_state=None):
    """Create appropriate cross-validation splitter.

    Parameters
    ----------
    cv : int
        Number of folds
    groups : array-like | None
        Group labels (for StratifiedGroupKFold)
    stratified : bool
        Whether to use stratified splitting
    random_state : int | None
        Random seed

    Returns
    -------
    splitter
        Cross-validation splitter instance such as
        `~~sklearn.model_selection.KFold` or
        `~~sklearn.model_selection.StratifiedGroupKFold`.
    """
    if groups is not None:
        if stratified:
            return StratifiedGroupKFold(n_splits=cv)
        else:
            return GroupKFold(n_splits=cv)
    else:
        if stratified:
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            return KFold(n_splits=cv, shuffle=True, random_state=random_state)
