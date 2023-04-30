# # ## Kernelized Support Vector Machines
#
#1.Classification
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)
from sklearn.datasets import make_classification, make_blobs
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8,
                       cluster_std = 1.3, random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)

# # The default SVC kernel is radial basis function (RBF)
# plot_class_regions_for_classifier(SVC().fit(X_train, y_train),
#                                  X_train, y_train, None, None,
#                                  'Support Vector Classifier: RBF kernel')
# plt.show()

# Compare decision boundries with polynomial kernel, degree = 3
# plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3)
#                                  .fit(X_train, y_train), X_train,
#                                  y_train, None, None,
#                                  'Support Vector Classifier: Polynomial kernel, degree = 3')
# plt.show()

##2.Support Vector Machine with RBF kernel: gamma parameter
# fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))
#
# for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
#     clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
#     title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
#     plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
#                                              None, None, title, subplot)
#     plt.tight_layout()
# plt.show()

##3.Support Vector Machine with RBF kernel: using both C and gamma parameter

# X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
# fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)
#
# for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
#
#     for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
#         title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
#         clf = SVC(kernel = 'rbf', gamma = this_gamma,
#                  C = this_C).fit(X_train, y_train)
#         plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
#                                                  X_test, y_test, title,
#                                                  subplot)
#         plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()

# # ### Application of SVMs to a real dataset: unnormalized data
# cancer = load_breast_cancer()
# (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
# X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
#                                                    random_state = 0)
# clf = SVC(C=10).fit(X_train, y_train)
# print('Breast cancer dataset (unnormalized features)')
# print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))

# # ### Application of SVMs to a real dataset: normalized data with feature preprocessing using minmax scaling
# cancer = load_breast_cancer()
# (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
# X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
#                                                    random_state = 0)
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# clf = SVC(C=10).fit(X_train_scaled, y_train)
# print('Breast cancer dataset (normalized with MinMax scaling)')
# print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
#      .format(clf.score(X_train_scaled, y_train)))
# print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
#      .format(clf.score(X_test_scaled, y_test)))
