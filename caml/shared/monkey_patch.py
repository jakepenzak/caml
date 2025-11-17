# Monkey patches for EconML scoring & validation utilities.
# This will be overhauled when we build out the new scoring & validation utilities.

import numpy as np
from econml.validate import DRTester


def patched_get_cate_preds(self, Xval, Xtrain):
    base = self.treatments[0]
    vals = [self.cate.effect(X=Xval, T0=base, T1=t) for t in self.treatments[1:]]
    self.cate_preds_val_ = np.stack(vals).T[0]

    if Xtrain is not None:
        trains = [
            self.cate.effect(X=Xtrain, T0=base, T1=t) for t in self.treatments[1:]
        ]
        self.cate_preds_train_ = np.stack(trains).T[0]


DRTester.get_cate_preds = patched_get_cate_preds  # pyright: ignore[reportAttributeAccessIssue]
