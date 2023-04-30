# # ## Linear models for regression
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

#1.Linear regression
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)
#
# print('linear model coeff (w): {}'
#      .format(linreg.coef_))
# print('linear model intercept (b): {:.3f}'
#      .format(linreg.intercept_))
# print('R-squared score (training): {:.3f}'
#      .format(linreg.score(X_train, y_train)))
# print('R-squared score (test): {:.3f}'
#      .format(linreg.score(X_test, y_test)))

##2.Linear regression: example plot

# plt.figure(figsize=(5,4))
# plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)
# plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
# plt.title('Least-squares linear regression')
# plt.xlabel('Feature value (x)')
# plt.ylabel('Target value (y)')
# plt.show()
#
# X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
#                                                    random_state = 0)
# linreg = LinearRegression().fit(X_train, y_train)
#
# print('Crime dataset')
# print('linear model intercept: {}'
#      .format(linreg.intercept_))
# print('linear model coeff:\n{}'
#      .format(linreg.coef_))
# print('R-squared score (training): {:.3f}'
#      .format(linreg.score(X_train, y_train)))
# print('R-squared score (test): {:.3f}'
#      .format(linreg.score(X_test, y_test)))

##3.Ridge regression
# from sklearn.linear_model import Ridge
# (X_crime, y_crime) = load_crime_dataset()
#
# X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
#
# linridge = Ridge(alpha=20.0).fit(X_train, y_train)
#
# print('Crime dataset')
# print('ridge regression linear model intercept: {}'
#      .format(linridge.intercept_))
# print('ridge regression linear model coeff:\n{}'
#      .format(linridge.coef_))
# print('R-squared score (training): {:.3f}'
#      .format(linridge.score(X_train, y_train)))
# print('R-squared score (test): {:.3f}'
#      .format(linridge.score(X_test, y_test)))
# print('Number of non-zero features: {}'
#      .format(np.sum(linridge.coef_ != 0)))

##3.Ridge regression with feature normalization

# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# (X_crime, y_crime) = load_crime_dataset()
#
# X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
#
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

# print('Crime dataset')
# print('ridge regression linear model intercept: {}'
#      .format(linridge.intercept_))
# print('ridge regression linear model coeff:\n{}'
#      .format(linridge.coef_))
# print('R-squared score (training): {:.3f}'
#      .format(linridge.score(X_train_scaled, y_train)))
# print('R-squared score (test): {:.3f}'
#      .format(linridge.score(X_test_scaled, y_test)))
# print('Number of non-zero features: {}'
#      .format(np.sum(linridge.coef_ != 0)))

##3.Ridge regression with regularization parameter: alpha

# print('Ridge regression: effect of alpha regularization parameter\n')
# for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
#     linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
#     r2_train = linridge.score(X_train_scaled, y_train)
#     r2_test = linridge.score(X_test_scaled, y_test)
#     num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
#     print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
#          .format(this_alpha, num_coeff_bigger, r2_train, r2_test))

##4.Lasso regression
# from sklearn.linear_model import Lasso
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# (X_crime, y_crime) = load_crime_dataset()
#
# X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
#                                                    random_state = 0)
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)
#
# print('Crime dataset')
# print('lasso regression linear model intercept: {}'
#      .format(linlasso.intercept_))
# print('lasso regression linear model coeff:\n{}'
#      .format(linlasso.coef_))
# print('Non-zero features: {}'
#      .format(np.sum(linlasso.coef_ != 0)))
# print('R-squared score (training): {:.3f}'
#      .format(linlasso.score(X_train_scaled, y_train)))
# print('R-squared score (test): {:.3f}\n'
#      .format(linlasso.score(X_test_scaled, y_test)))
# print('Features with non-zero weight (sorted by absolute magnitude):')
#
# for e in sorted (list(zip(list(X_crime), linlasso.coef_)),
#                 key = lambda e: -abs(e[1])):
#     if e[1] != 0:
#         print('\t{}, {:.3f}'.format(e[0], e[1]))

##4.Lasso regression with regularization parameter: alpha
# print('Lasso regression: effect of alpha regularization\nparameter on number of features kept in final model\n')
#
# for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
#     linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
#     r2_train = linlasso.score(X_train_scaled, y_train)
#     r2_test = linlasso.score(X_test_scaled, y_test)
#
#     print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
#          .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))

##5. Polynomial regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
#
X_F1, y_F1 = make_friedman1(n_samples = 100,
                           n_features = 7, random_state=0)

#
# X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1,
#                                                    random_state = 0)
# linreg = LinearRegression().fit(X_train, y_train)
#
# print('linear model coeff (w): {}'
#      .format(linreg.coef_))
# print('linear model intercept (b): {:.3f}'
#      .format(linreg.intercept_))
# print('R-squared score (training): {:.3f}'
#      .format(linreg.score(X_train, y_train)))
# print('R-squared score (test): {:.3f}'
#      .format(linreg.score(X_test, y_test)))
#
print('\nNow we transform the original input data to add\npolynomial features up to degree 2 (quadratic)\n')
poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('(poly deg 2) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.3f}\n'
     .format(linreg.score(X_test, y_test)))
#
# print('\nAddition of many polynomial features often leads to\noverfitting,
#so we often use polynomial features in combination\nwith regression that
#has a regularization penalty, like ridge\nregression.\n')
#
# X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
#                                                    random_state = 0)
# linreg = Ridge().fit(X_train, y_train)
#
# print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
#      .format(linreg.coef_))
# print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
#      .format(linreg.intercept_))
# print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
#      .format(linreg.score(X_train, y_train)))
# print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
#      .format(linreg.score(X_test, y_test)))
