# # Assignment 3 - Evaluation
# In this assignment you will train several models and evaluate
#how effectively they predict instances of fraud using data based
#on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
# Each row in `fraud_data.csv` corresponds to a credit card transaction.
#Features include confidential variables `V1` through `V28` as well as `Amount`
#which is the amount of the transaction. 

# The target is stored in the `class` column, where a value of 1 corresponds
#to an instance of fraud and 0 corresponds to an instance of not fraud.

import numpy as np
import pandas as pd
df = pd.read_csv('fraud_data.csv')

# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the
#observations in the dataset are instances of fraud (1)?

# *This function should return a float between 0 and 1.*
def answer_one():
    ans= df['Class'].where(df['Class']==1).count()/len(df['Class'])
    print(ans)
    # Your code here
    return ans # Return your answer
#answer_one()

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# ### Question 2
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above),
#train a dummy classifier that classifies everything as the majority class of
#the training data. What is the accuracy of this classifier? What is the recall?

# *This function should a return a tuple with two floats, i.e.
#`(accuracy score, recall score)`.*

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_predictions = dummy_majority.predict(X_test)
    #print(np.unique(y_dummy_predictions))
    score = dummy_majority.score(X_test, y_test)

    recall= recall_score(y_test, y_dummy_predictions)

    print('Score: ', score ,'Recall: ', recall)
    # print('Precision: ', precision_score(y_test, y_dummy_predictions))
    ans=(score, recall)
    return ans
#answer_two()


# ### Question 3
# Using X_train, X_test, y_train, y_test (as defined above),
#train a SVC classifer using the default parameters. What is the accuracy,
#recall, and precision of this classifier?
# *This function should a return a tuple with three floats, i.e.
# `(accuracy score, recall score, precision score)`.*

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    svc_classifier = SVC(kernel='rbf', C=1).fit(X_train, y_train)
    predicitons=svc_classifier.predict(X_test)
    svc_score=svc_classifier.score(X_test, y_test) #accuracy
    svc_recall=recall_score(y_test, predicitons) #recall
    svc_precision=precision_score(y_test, y_test) #precision
    # print('SVM Score ', svc_score)
    # print('SVM Recall ', svc_recall)
    # print('SVM Precision ', svc_precision)
    ans=(svc_score, svc_recall, svc_precision)
    print(ans)
    return ans
# answer_three()

# ### Question 4
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`,
#what is the confusion matrix when using a threshold of -220 on the
#decision function. Use X_test and y_test.
# *This function should return a confusion matrix,
# a 2x2 numpy array with 4 integers.*

# print(type(y_test))
# print(y_test.head(20))

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    svc_classifier = SVC(kernel='rbf', C=1e9, gamma=1e-07).fit(X_train, y_train)
    #y_scores= numpy.ndarray, predictions. Da las distancias de
    #cada punto con respecto al modelo creado. Si es negativa la distancia,
    #el punto esta por debajo del modelo, si es positiva, esta por encima.
    y_scores=svc_classifier.decision_function(X_test)
    y_scores = np.where(y_scores > (-220), 1, 0) #convertimos las distancoas a 1 si son (+),
    #0 si son (-), desde el punto -220 (trheshold)
    confusion = confusion_matrix(y_test, y_scores)

    print(confusion)
    return confusion# Return your answer
# answer_four()


# ### Question 5
# Train a logisitic regression classifier with default parameters using
#X_train and y_train.
# For the logisitic regression classifier, create a precision recall curve
#and a roc curve using y_test and the probability estimates
#for X_test (probability it is fraud).

# Looking at the precision recall curve, what is the recall when
#the precision is `0.75`?
# Looking at the roc curve, what is the true positive rate when
#the false positive rate is `0.16`?
# *This function should return a tuple with two floats, i.e.
#`(recall, true positive rate)`.*

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # ##Precision Recall curve
    lr = LogisticRegression().fit(X_train, y_train)
    # predictions = lr.predict(X_test)
    # precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    # # print(precision)
    # # print(recall)
    # # print(thresholds)
    # closest_zero = np.argmin(np.abs(thresholds))
    # closest_zero_p = precision[closest_zero]
    # closest_zero_r = recall[closest_zero]
    # # print(closest_zero)
    # # print(closest_zero_p)
    # # print(closest_zero_r)
    # plt.figure(figsize=(9,5))
    # plt.xlim([0.0, 1.01])
    # plt.ylim([0.0, 1.01])
    # plt.plot(precision, recall, label='Precision-Recall Curve')
    # plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    # plt.xlabel('Precision', fontsize=16)
    # plt.ylabel('Recall', fontsize=16)
    # plt.grid(visible='True')
    # plt.show()

    ## ROC Curve
    y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.figure(figsize=(9,5))
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.grid(visible='True')
    plt.show()

    ans=(0.83, 0.93)
    return ans# Return your answer
answer_five()

# ### Question 6
# Perform a grid search over the parameters listed below for a
#Logisitic Regression classifier, using recall for scoring and the
#default 3-fold cross validation.
# `'penalty': ['l1', 'l2']`
# `'C':[0.01, 0.1, 1, 10, 100]`
# From `.cv_results_`, create an array of the mean test scores of
#each parameter combination. i.e.

# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|

# *This function should return a 5 by 2 numpy array with 10 floats.*
# *Note: do not return a DataFrame, just the values denoted by '?' above
#in a numpy array. You might need to reshape your raw result to meet
#the format we are looking for.*

def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression().fit(X_train, y_train)
    # lr_predicted = lr.predict(X_test)
    grid_values = {'penalty': ['l1', 'l2'], 'C':[0.01, 0.1, 1, 10, 100] }
    grid_clf= GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
    grid_clf.fit(X_train, y_train)

    ans=grid_clf.cv_results_['mean_test_score'].reshape(5,2) #numpy.ndarray

    print(ans)
    return ans
# answer_six()

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    get_ipython().magic('matplotlib notebook')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())
