#Linear classifiers: support vector machines
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import load_breast_cancer

##1.Linear Support Vector Machine
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)

# X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
#
# fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
# this_C = 1.0
# clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
# title = 'Linear SVC, C = {:.3f}'.format(this_C)
# plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)
# plt.show()

##2.Linear Support Vector Machine: C parameter

# X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
# fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))
#
# for this_C, subplot in zip([0.00001, 100], subaxes):
#     clf = LinearSVC(C=this_C).fit(X_train, y_train)
#     title = 'Linear SVC, C = {:.5f}'.format(this_C)
#     plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
#                                              None, None, title, subplot)
# plt.tight_layout()
# plt.show()

##3.Application to real dataset
# cancer = load_breast_cancer()
# (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
# X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
#
# clf = LinearSVC().fit(X_train, y_train)
# print('Breast cancer dataset')
# print('Accuracy of Linear SVC classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Linear SVC classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))