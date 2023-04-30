# ### Two-feature classification example using the digits dataset

# #### Optimizing a classifier using different evaluation metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from adspy_shared_utilities import plot_class_regions_for_classifier

# dataset = load_digits()
# X, y = dataset.data, dataset.target == 1
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# # Create a two-feature input vector matching the example plot above
# # We jitter the points (add a small amount of random noise) in case there are areas
# # in feature space where many instances have the same features.
# jitter_delta = 0.25
# X_twovar_train = X_train[:,[20,59]]+ np.random.rand(X_train.shape[0], 2) - jitter_delta
# X_twovar_test  = X_test[:,[20,59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta
#
# clf = SVC(kernel = 'linear').fit(X_twovar_train, y_train)
# grid_values = {'class_weight':['balanced', {1:2},{1:3},{1:4},{1:5},{1:10},{1:20},{1:50}]}
# plt.figure(figsize=(9,6))
# for i, eval_metric in enumerate(('precision','recall', 'f1','roc_auc')):
#     grid_clf_custom = GridSearchCV(clf, param_grid=grid_values, scoring=eval_metric)
#     grid_clf_custom.fit(X_twovar_train, y_train)
#     print('Grid best parameter (max. {0}): {1}'
#           .format(eval_metric, grid_clf_custom.best_params_))
#     print('Grid best score ({0}): {1}'
#           .format(eval_metric, grid_clf_custom.best_score_))
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
#     plot_class_regions_for_classifier_subplot(grid_clf_custom, X_twovar_test, y_test, None,
#                                              None, None,  plt.subplot(2, 2, i+1))
#
#     plt.title(eval_metric+'-oriented SVC')
# plt.tight_layout()
# plt.show()


# #### Precision-recall curve for the default SVC classifier (with balanced class weights)

dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# create a two-feature input vector matching the example plot above
jitter_delta = 0.25
X_twovar_train = X_train[:,[20,59]]+ np.random.rand(X_train.shape[0], 2) - jitter_delta
X_twovar_test  = X_test[:,[20,59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta

clf = SVC(kernel='linear', class_weight='balanced').fit(X_twovar_train, y_train)

y_scores = clf.decision_function(X_twovar_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]


plot_class_regions_for_classifier(clf, X_twovar_test, y_test, title="SVC, class_weight = 'balanced', optimized for accuracy")

plt.figure(figsize=(6,6))
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.title ("Precision-recall curve: SVC, class_weight = 'balanced'")
plt.plot(precision, recall, label = 'Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle='none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.grid(visible='True')
plt.show()
print('At zero threshold, precision: {:.2f}, recall: {:.2f}'
      .format(closest_zero_p, closest_zero_r))
