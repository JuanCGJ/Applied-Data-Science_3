# ## Ensembles of Decision Trees
# ### Random forests
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import make_classification, make_blobs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

# X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2,
#                                                    random_state = 0)
# fig, subaxes = plt.subplots(1, 1, figsize=(8, 6))
#
# clf = RandomForestClassifier().fit(X_train, y_train)
# title = 'Random Forest Classifier, complex binary dataset, default settings'
# plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test,
#                                          y_test, title, subaxes)
#
# plt.show()

##fruits dataset
fruits = pd.read_table('fruit_data_with_colors.txt')
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']
# X_train, X_test, y_train, y_test = train_test_split(X_fruits.to_numpy(),
#                                                    y_fruits.to_numpy(),
#                                                    random_state = 0)
# fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))
#
# title = 'Random Forest, fruits dataset, default settings'
# pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
#
# for pair, axis in zip(pair_list, subaxes):
#     X = X_train[:, pair]
#     y = y_train
#
#     clf = RandomForestClassifier().fit(X, y)
#     plot_class_regions_for_classifier_subplot(clf, X, y, None,
#                                              None, title, axis,
#                                              target_names_fruits)
#
#     axis.set_xlabel(feature_names_fruits[pair[0]])
#     axis.set_ylabel(feature_names_fruits[pair[1]])
#
# plt.tight_layout()
# plt.show()
#
# clf = RandomForestClassifier(n_estimators = 10,
#                             random_state=0).fit(X_train, y_train)
#
# print('Random Forest, Fruit dataset, default settings')
# print('Accuracy of RF classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of RF classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))


# #### Random Forests on a real-world dataset
# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
# X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
#
# clf = RandomForestClassifier(max_features = 8, random_state = 0)
# clf.fit(X_train, y_train)
#
# print('Breast cancer dataset')
# print('Accuracy of RF classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of RF classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))


# # ### Gradient-boosted decision trees

# X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
# fig, subaxes = plt.subplots(1, 1, figsize=(8, 8))
#
# clf = GradientBoostingClassifier().fit(X_train, y_train)
# title = 'GBDT, complex binary dataset, default settings'
# plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test,
#                                          y_test, title, subaxes)
# plt.show()


# # #### Gradient boosted decision trees on the fruit dataset
# title = 'GBDT, complex binary dataset, default settings'
# X_train, X_test, y_train, y_test = train_test_split(X_fruits.to_numpy(),
#                                                    y_fruits.to_numpy(),
#                                                    random_state = 0)
# fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))
#
# pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
#
# for pair, axis in zip(pair_list, subaxes):
#     X = X_train[:, pair]
#     y = y_train
#
#     clf = GradientBoostingClassifier().fit(X, y)
#     plot_class_regions_for_classifier_subplot(clf, X, y, None,
#                                              None, title, axis,
#                                              target_names_fruits)
#
#     axis.set_xlabel(feature_names_fruits[pair[0]])
#     axis.set_ylabel(feature_names_fruits[pair[1]])
#
# plt.tight_layout()
# plt.show()
# clf = GradientBoostingClassifier().fit(X_train, y_train)
#
# print('GBDT, Fruit dataset, default settings')
# print('Accuracy of GBDT classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of GBDT classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))


# # #### Gradient-boosted decision trees on a real-world dataset
# X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
#
# clf = GradientBoostingClassifier(random_state = 0)
# clf.fit(X_train, y_train)
#
# print('Breast cancer dataset (learning_rate=0.1, max_depth=3)')
# print('Accuracy of GBDT classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of GBDT classifier on test set: {:.2f}\n'
#      .format(clf.score(X_test, y_test)))
#
# clf = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0)
# clf.fit(X_train, y_train)
#
# print('Breast cancer dataset (learning_rate=0.01, max_depth=2)')
# print('Accuracy of GBDT classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of GBDT classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))
