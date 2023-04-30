
# # #### Logistic regression for binary classification on fruits dataset using height, width features (positive class: apple, negative class: others)
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset
from adspy_shared_utilities import plot_two_class_knn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)

np.set_printoptions(precision=2)
fruits = pd.read_table('fruit_data_with_colors.txt')
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']
y_fruits_apple = y_fruits_2d == 1
print(y_fruits_apple.shape)

##1.Logistic regression

# fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
# y_fruits_apple = y_fruits_2d == 1   # make into a binary problem: apples vs everything else
# X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d.to_numpy(),
#                 y_fruits_apple.to_numpy(), random_state = 0)
#
# clf = LogisticRegression(C=100).fit(X_train, y_train)
# plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None,
#                                          None, 'Logistic regression \
# for binary classification\nFruit dataset: Apple vs others', subaxes)
# h = 6
# w = 8
# print('A fruit with height {} and width {} is predicted to be: {}'
#      .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
#
# h = 10
# w = 7
# print('A fruit with height {} and width {} is predicted to be: {}'
#      .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
# subaxes.set_xlabel('height')
# subaxes.set_ylabel('width')
# plt.show()
#
# print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))

##2.Logistic regression on simple synthetic dataset
# X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
#                                 n_redundant=0, n_informative=2,
#                                 n_clusters_per_class=1, flip_y = 0.1,
#                                 class_sep = 0.5, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2,
#                                                    random_state = 0)
#
# fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
# clf = LogisticRegression().fit(X_train, y_train)
# title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(1.0)
# plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
#                                          None, None, title, subaxes)
# plt.show()
# print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))

#3.Logistic regression regularization: C parameter

# X_train, X_test, y_train, y_test = (
# train_test_split(X_fruits_2d.to_numpy(),
#                 y_fruits_apple.to_numpy(),
#                 random_state=0))
#
# fig, subaxes = plt.subplots(3, 1, figsize=(5, 70))
#
# for this_C, subplot in zip([0.1, 1, 100], subaxes):
#     clf = LogisticRegression(C=this_C).fit(X_train, y_train)
#     title ='Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)
#
#     plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
#                                              X_test, y_test, title,
#                                              subplot)
# plt.tight_layout()
# plt.show()

##4.Application to real dataset
# cancer = load_breast_cancer()
# (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
#
# X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
#
# clf = LogisticRegression().fit(X_train, y_train)
# print('Breast cancer dataset')
# print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))
