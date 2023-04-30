# ### Regression evaluation metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.datasets import load_digits

diabetes = datasets.load_diabetes()

X = diabetes.data[:, None, 6]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lm = LinearRegression().fit(X_train, y_train)
lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)

y_predict = lm.predict(X_test)
y_predict_dummy_mean = lm_dummy_mean.predict(X_test)

# print('Linear model, coefficients: ', lm.coef_)
# print("Mean squared error (dummy): {:.2f}".format(mean_squared_error(y_test,
#                                                                      y_predict_dummy_mean)))
# print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))
# print("r2_score (dummy): {:.2f}".format(r2_score(y_test, y_predict_dummy_mean)))
# print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))
#
# # Plot outputs
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, y_predict, color='green', linewidth=2)
# plt.plot(X_test, y_predict_dummy_mean, color='red', linestyle = 'dashed',
#          linewidth=2, label = 'dummy')
#
# plt.show()


# # ### Model selection using evaluation metrics
# # #### Cross-validation example
#
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC
#
# dataset = load_digits()
# # again, making this a binary problem with 'digit 1' as positive class
# # and 'not 1' as negative class
# X, y = dataset.data, dataset.target == 1
# clf = SVC(kernel='linear', C=1)
#
# # accuracy is the default scoring metric
# print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5))
# # use AUC as scoring metric
# print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc'))
# # use recall as scoring metric
# print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=5, scoring = 'recall'))

#
# # #### Grid search example
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test)

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

# #### Evaluation metrics supported for model selection
# from sklearn.metrics.scorer import SCORERS
# print(sorted(list(SCORERS.keys())))
