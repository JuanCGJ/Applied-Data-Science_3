import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
# print(cancer.keys())
# print(cancer.data)
# print(cancer.feature_names)
# print(cancer.target)

def answer_zero():
    l=len(cancer['feature_names'])
    # return print(l)
answer_zero()

def answer_one():
    data=cancer.data
    cols=cancer.feature_names
    df = pd.DataFrame(data, columns=cols)
    df['target']=cancer.target
    # print(df.index)
    # print(df)
    return df
answer_one()

def answer_two():
    cancerdf = answer_one()
    # print(cancerdf)
    d=cancerdf['target'].unique()
    target = pd.Series(data=d, index =['malignant', 'benign'])
    # print(target)
    # return target
answer_two()

def answer_three():
    cancerdf = answer_one()
    # print(cancerdf)
    X = cancerdf.iloc[:,0:30]
    y = cancerdf.iloc[:,30]
    # print(X.shape)
    # print(y.shape)
    return X, y
answer_three()


def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # return print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    return X_train, X_test, y_train, y_test
answer_four()


def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    fit_estimator = knn.fit(X_train, y_train)
    # print(type(fit_estimator))
    return knn
answer_five()

def answer_six():
    knn =answer_five()
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    # print(means)
    prediction = knn.predict(means)
    #return print(prediction)
answer_six()

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    prediction = knn.predict(X_test)
    # print(type(prediction))
    # return print(prediction)
answer_seven()

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    # print('Xtest', X_test.shape, 'y_test', y_test.shape)
    mean= knn.score(X_test, y_test)
    return print(mean)
answer_eight()

def accuracy_plot():

    X_train, X_test, y_train, y_test = answer_four()
    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]
    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2),
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()
accuracy_plot()
