# # ### Multi-class classification with linear models
##1.LinearSVC with M classes generates M one vs rest classifiers.
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
from sklearn.svm import LinearSVC

fruits = pd.read_table('fruit_data_with_colors.txt')
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']
y_fruits_apple = y_fruits_2d == 1

# X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)
#
# clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
# print('Coefficients:\n', clf.coef_)
# print('Intercepts:\n', clf.intercept_)

##2.Multi-class results on the fruit dataset

# plt.figure(figsize=(6,6))
# colors = ['r', 'g', 'b', 'y']
# cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])
#
# X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)
# clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
# plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
#            c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)
#
# x_0_range = np.linspace(-10, 15)
#
# for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
#     # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b,
#     # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a
#     # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
#     plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
#
# plt.legend(target_names_fruits)
# plt.xlabel('height')
# plt.ylabel('width')
# plt.xlim(-2, 12)
# plt.ylim(-2, 15)
# plt.show()
