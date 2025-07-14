# Monkey patches for EconML scoring & validation utilities.
# This will be overhauled when we build out the new scoring & validation utilities.

from econml.validate import DRTester
from econml.validate.utils import calculate_dr_outcomes
from sklearn.model_selection import cross_val_predict


# def patched_fit_nuisance_cv(self, X, D, y):
#     splits = self.get_cv_splits([X], D)
#     self.model_propensity.enable_categorical = True
#     self.model_regression.enable_categorical = True
#     prop_preds = cross_val_predict(
#         self.model_propensity, X, D.squeeze(), cv=splits, method="predict_proba"
#     )
#     # Predict outcomes
#     # T-learner logic
#     N = X.shape[0]
#     reg_preds = np.zeros((N, self.n_treat + 1))
#     for k in range(self.n_treat + 1):
#         for train, test in splits:
#             # Create boolean mask for treatment group
#             treat_mask = D.iloc[train].squeeze() == self.treatments[k]
#             X_train_k = X.iloc[train][treat_mask]
#             y_train_k = y.iloc[train][treat_mask]
#             model_regression_fitted = self.model_regression.fit(X_train_k, y_train_k)
#             reg_preds[test, k] = model_regression_fitted.predict(X.iloc[test])

#     return reg_preds, prop_preds


# def patched_fit_nuisance_train(self, Xtrain, Dtrain, ytrain, Xval):
#     self.model_propensity.enable_categorical = True
#     self.model_regression.enable_categorical = True
#     # Fit propensity in treatment
#     model_propensity_fitted = self.model_propensity.fit(Xtrain, Dtrain)
#     # Predict propensity scores
#     prop_preds = model_propensity_fitted.predict_proba(Xval)

#     # Possible treatments (need to allow more than 2)
#     n = Xval.shape[0]
#     reg_preds = np.zeros((n, self.n_treat + 1))
#     for i in range(self.n_treat + 1):
#         treat_mask = Dtrain.squeeze() == self.treatments[i]
#         model_regression_fitted = self.model_regression.fit(
#             Xtrain[treat_mask],
#             ytrain[treat_mask],
#         )
#         reg_preds[:, i] = model_regression_fitted.predict(Xval)

#     return reg_preds, prop_preds


# def patched_fit_nuisance(
#     self,
#     Xval,
#     Dval,
#     yval,
#     Xtrain,
#     Dtrain,
#     ytrain,
# ):
#     self.Dval = Dval

#     # Unique treatments (ordered, includes control)
#     self.treatments = np.sort(np.unique(Dval))

#     # Number of treatments (excluding control)
#     self.n_treat = len(self.treatments) - 1

#     # Indicator for whether
#     self.fit_on_train = (
#         (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None)
#     )

#     if self.fit_on_train:
#         # Get DR outcomes in training sample
#         reg_preds_train, prop_preds_train = self.fit_nuisance_cv(Xtrain, Dtrain, ytrain)
#         self.dr_train_ = calculate_dr_outcomes(
#             Dtrain.to_numpy(),
#             ytrain.to_numpy(),
#             reg_preds_train,
#             prop_preds_train,
#         )

#         # Get DR outcomes in validation sample
#         reg_preds_val, prop_preds_val = self.fit_nuisance_train(
#             Xtrain, Dtrain, ytrain, Xval
#         )
#         self.dr_val_ = calculate_dr_outcomes(
#             Dval.to_numpy(),
#             yval.to_numpy(),
#             reg_preds_val,
#             prop_preds_val,
#         )

#     else:
#         # Get DR outcomes in validation sample
#         reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
#         self.dr_val_ = calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)

#     # Calculate ATE in the validation sample
#     self.ate_val = self.dr_val_.mean(axis=0)

#     return self


# def patched_get_cate_preds(self, Xval, Xtrain):
#     base = self.treatments[0]
#     vals = [self.cate.effect(X=Xval, T0=base, T1=t) for t in self.treatments[1:]]
#     self.cate_preds_val_ = np.stack(vals).T
#     print(len(self.treatments))
#     if len(self.treatments) == 2:
#         self.cate_preds_val_ = self.cate_preds_val_.reshape(-1, 1)

#     if Xtrain is not None:
#         trains = [
#             self.cate.effect(X=Xtrain, T0=base, T1=t) for t in self.treatments[1:]
#         ]
#         self.cate_preds_train_ = np.stack(trains).T
#         if len(self.treatments) == 2:
#             self.cate_preds_train_ = self.cate_preds_train_.reshape(-1, 1)


# DRTester.fit_nuisance_cv = patched_fit_nuisance_cv
# DRTester.fit_nuisance_train = patched_fit_nuisance_train
# DRTester.fit_nuisance = patched_fit_nuisance  # pyright: ignore[reportAttributeAccessIssue]
# DRTester.get_cate_preds = patched_get_cate_preds  # pyright: ignore[reportAttributeAccessIssue]
