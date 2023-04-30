# # ## Cross-validation
##1.Example based on k-NN classifier with fruit dataset (2 features)
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

np.set_printoptions(precision=2)
fruits = pd.read_table('fruit_data_with_colors.txt')
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']
y_fruits_apple = y_fruits_2d == 1

clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits_2d.to_numpy()
y = y_fruits_2d.to_numpy()
cv_scores = cross_val_score(clf, X, y)

print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))


# ### A note on performing cross-validation for more advanced scenarios.
# In some cases (e.g. when feature values have very different ranges),
#we've seen the need to scale or normalize the training and test sets before use
#with a classifier. The proper way to do cross-validation when you need to scale the
#data is *not* to scale the entire dataset with a single transform,
#since this will indirectly leak information into the training data
# about the whole dataset, including the test data
#(see the lecture on data leakage later in the course).
#Instead, scaling/normalizing must be computed and applied for each
#cross-validation fold separately.  To do this, the easiest way in scikit-learn
#is to use *pipelines*.  While these are beyond the scope of this course,
#further information is available in the scikit-learn documentation here:
# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# or the Pipeline section in the recommended textbook:
#Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido (O'Reilly Media).

#2.Validation curve example
param_range = np.logspace(-3, 3, 4)
train_scores, test_scores = validation_curve(SVC(), X, y,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)
print(train_scores)

print(test_scores)

# This code based on scikit-learn validation_plot example
#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
# plt.figure()
#
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$ (gamma)')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
plt.show()
