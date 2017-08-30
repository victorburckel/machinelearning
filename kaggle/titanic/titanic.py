# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:46:08 2017

@author: Victor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import pipeline

def plot_learning_curve(estimator, X, y):
    plt.figure()
    plt.title('Learning curve')
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = model_selection .learning_curve(estimator, X, y, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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
    
    # Plot the learning curve
    plot_learning_curve(estimator.best_estimator_, X_train, y_train)
    
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