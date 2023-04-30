import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def blight_model():

    df=pd.read_csv('train.csv',encoding = "latin1")
    #print(df.shape)
    dftrain_dummies = pd.get_dummies(df, columns=['agency_name'])
    # print(dftrain_dummies.shape)
    data_health = dftrain_dummies.loc[dftrain_dummies['agency_name_Health Department'] == 1].index
    #print(data_health.shape)# 8903
    data_neigh=   dftrain_dummies.loc[dftrain_dummies['agency_name_Neighborhood City Halls'] == 1].index
    #print(data_neigh.shape)# 2
    dftrain_dummies.drop(data_health, inplace=True)
    dftrain_dummies.drop(data_neigh, inplace=True)
    #elimino rows con health and neigh == 1, no los utilizare en el train
    #ya que el test no los tiene en cuenta
    # print(dftrain_dummies.shape)

    # for e,x in enumerate(dftrain_dummies.columns):
    #     print(e, x)
    X_train=dftrain_dummies.iloc[:,[0,18,19,20,21,24,33,34,35]].fillna(0)
    # print(X_train.columns)
    y_train=dftrain_dummies.iloc[:,32].fillna(0)

    df2=pd.read_csv('test.csv',encoding = "latin1")
    # print(df2.shape)
    dftest_dummies = pd.get_dummies(df2, columns=['agency_name'])
    # print(dftest_dummies.shape)
    # for j,k in enumerate(dftest_dummies.columns):
    #     print(j, k)

    X_test=dftest_dummies.iloc[:,[0,18,19,20,21,24,26,27,28]]


    # print(X_test.columns)
    # print(X_train.columns == X_test.columns, '\n')
    # print(X_train.dtypes)
    # print('\n', X_test.dtypes)

    # print('\n X_train valores null por columna: ', X_train.isnull().sum())
    # print('\n y_train cantidad de valores null: ', y_train.isnull().sum())
    # print('\n y_train valores null: ', y_train.isnull(),'\n')
    # print('X_test valores null por columna\n', X_test.isnull().sum())

    # lr = LogisticRegression().fit(X_train, y_train)
    # y_score_lr = lr.fit(X_train, y_train).predict_proba(X_test)#probabilities (Menor,mayor)

    clf = RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state=0).fit(X_train, y_train)
    y_score_lr=clf.predict_proba(X_test)

    proba=y_score_lr[:,1]
    ticketids=X_test['ticket_id']
    ans=pd.DataFrame(data=proba, index=ticketids)
    return ans
blight_model()
