# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:10:14 2020

@author: AJINKYA
"""

# Importing Libraries
import pandas as pd
import numpy as np

# Importing Training Data
train = pd.read_csv("F:/python downloads/hackathon dataset/TRAINING.csv")

#replacing values of yes & no by 1,0
train.at[train['roof']=='NO', 'roof'] = 0
train.at[train['roof']=='YES', 'roof'] = 1

#factorizing variables
train['Grade'], _ = pd.factorize(train['Grade'], sort=True)
train['roof'], _ = pd.factorize(train['roof'], sort=True)
train['roof']=train['roof'].replace(-1, np.nan)

# Finding out the total count of missing values
train.columns[train.isnull().any()].tolist()
train.isna().sum()

# imputing missing values in training data
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=1, weights="uniform")
train=imputer.fit_transform(train)
train=pd.DataFrame(train)

#loading test data and checking for any missing values
test = pd.read_csv("F:/python downloads/hackathon dataset/TEST.csv")

#removing special character '$' from expected price and converting it to astype integer.
test['EXPECTED']=test['EXPECTED'].map(lambda x:x.rstrip('$'))
test['EXPECTED'].astype(int)

#replacing values of yes and no by 1,0
test.at[test['roof']=='NO', 'roof'] = 0
test.at[test['roof']=='no', 'roof'] = 0
test.at[test['roof']=='YES', 'roof'] = 1
test.at[test['roof']=='yes', 'roof'] = 1

#factorizing test variables
test['roof'], _ = pd.factorize(test['roof'], sort=True)
test['roof']=test['roof'].replace(-1, np.nan)

# Finding out the total count of missing values
test.columns[test.isnull().any()].tolist()
test.isna().sum()

# imputing missing values in test data
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=1, weights="uniform")
test=imputer.fit_transform(test)
test=pd.DataFrame(test)
test = pd.DataFrame(test)

#seperating dependent and independent variable 
x = train.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,13]].values
y = train.iloc[:,12].values

#fitting model predicting test values using rbf
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x,y)

test=test.iloc[:,1:].values
predictions = classifier.predict(test)

#applying k-fold cross validation.  
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=x, y=y,cv=10)
accuracies.mean()
accuracies.std()

#applying grid search to find the best fit model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1, 10, 100, 1000], 'kernel': ['linear']},
             {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
             {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10,
                          n_jobs = -1)

grid_search = grid_search.fit(x, y)