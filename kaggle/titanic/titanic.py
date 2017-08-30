# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:46:08 2017

@author: Victor
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import pipeline


if __name__ == '__main__':
    # Read the training dataset
    trainDataSet = pd.read_table('train.csv', encoding='utf-8', sep=',', index_col=0, header=0)
    
    # Fills in missing values with average value
    trainDataSet['Embarked'].fillna('Unknwon', inplace = True)
    trainDataSet['Age'].fillna(30, inplace = True)
    
    # Generate dummy features from categorical features
    X = pd.get_dummies(trainDataSet[['Age','SibSp','Parch','Pclass','Sex','Embarked']], columns = ['Pclass','Sex','Embarked'])
    y = trainDataSet.Survived
    
    # Split the training set in training and test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    
    #Generate a pipeline that scales and applies logistic regression
    pipe = pipeline.Pipeline(steps=[('scaler', preprocessing.StandardScaler()), ('logistic', linear_model.LogisticRegression())])
    
    # Find the best value for parameter C
    estimator = model_selection.GridSearchCV(pipe, dict(logistic__C=np.logspace(-4, 4, 9)))
    estimator.fit(X_train, y_train)
    print(estimator.best_params_)
    
    # Estimate the prediction on the test dataset
    y_test_est = estimator.best_estimator_.predict(X_test)
    print(metrics.f1_score(y_test, y_test_est))
    print(metrics.accuracy_score(y_test, y_test_est))

    # Read the dataset to predict
    predictDataSet = pd.read_table('test.csv', encoding='utf-8', sep=',', index_col=0, header=0)
    predictDataSet['Embarked'].fillna('Unknwon', inplace = True)
    predictDataSet['Age'].fillna(30, inplace = True)
    
    # Generate features and predict output
    X_pred = pd.get_dummies(predictDataSet[['Age','SibSp','Parch','Pclass','Sex','Embarked']], columns = ['Pclass','Sex','Embarked'])
    X_pred = X_pred.reindex(columns = X.columns, fill_value=0)
    y_pred = estimator.best_estimator_.predict(X_pred)
    
    # Save output to format for kaggle submission
    np.savetxt('prediction.csv', np.column_stack((X_pred.index.values, y_pred)), delimiter=',', header = 'PassengerId,Survived', fmt = '%i', comments='')