# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:46:08 2017

@author: Victor
"""

from pandas import read_table
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

def read_data(filename, columns):
    '''
    Read the training / test data set
    '''
    
    dataSet = read_table(
            filename, 
            encoding='latin-1',
            sep=',',
            skipinitialspace=True,
            index_col=0,
            usecols = ['PassengerId'] + columns,
            header=0)
    
    #dataSet.dropna(inplace = True)
    dataSet['Embarked'].fillna('Unknwon', inplace = True)
    dataSet['Age'].fillna(30, inplace = True)
    
    return dataSet

def encode_categorical_features(trainDataSet, testDataSet, categoricalColumns):
    '''
    Encode categorical features in numerical features
    '''
    
    for categoricalColumn in categoricalColumns:
        le = preprocessing.LabelEncoder()
        trainDataSet[categoricalColumn] = le.fit_transform(trainDataSet[categoricalColumn])
        testDataSet[categoricalColumn] = le.transform(testDataSet[categoricalColumn])

def split_data_sets(trainDataSet, testDataSet, featuresColumns, predictionColumn):
    '''
    Convert training data set X, y
    Convert test data set to X_test
    Splits the test data set in training and validation set
    '''
    
    X = np.array(trainDataSet[featuresColumns], dtype=np.float)
    y = np.array(trainDataSet[predictionColumn], dtype=np.float)
    X_test = np.array(testDataSet, dtype=np.float)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.3)
    return X_train, X_val, y_train, y_val, X_test

def normalize_features(X_train, X_val, X_test):
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test

def train(X_train, X_val, y_train, y_val):
    classifier = linear_model.LogisticRegression(C=1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)
    print(metrics.f1_score(y_val, y_pred))
    print(metrics.accuracy_score(y_val, y_pred))
    
    y_test = classifier.predict(X_test)
    return y_test

if __name__ == '__main__':
    trainingDataSetFile = 'train.csv'
    testDataSetFile = 'test.csv'
    featuresColumns = ['Pclass','Sex','Age','SibSp','Parch','Embarked']
    predictionColumn = 'Survived'
    categoricalColumns = ['Pclass','Sex','Embarked']
    outputFile = 'prediction.csv'
    
    print('Reading training data set from %s' % trainingDataSetFile)
    trainDataSet = read_data(trainingDataSetFile, featuresColumns + [predictionColumn])
    
    print('Reading test data set from %s' % testDataSetFile)
    testDataSet = read_data(testDataSetFile, featuresColumns)
    
    print('Encoding categorical features')
    encode_categorical_features(trainDataSet, testDataSet, categoricalColumns)
    
    print('Splitting data sets')
    X_train, X_val, y_train, y_val, X_test = split_data_sets(trainDataSet, testDataSet, featuresColumns, predictionColumn)
    
    print('Normalizing features')
    normalize_features(X_train, X_val, X_test)
    
    print('Training classifier')
    y_test = train(X_train, X_val, y_train, y_val)
    
    np.savetxt(outputFile, np.column_stack((testDataSet.index.values, y_test)), delimiter=',', header = 'PassengerId,Survived', fmt = '%i', comments='')
    
