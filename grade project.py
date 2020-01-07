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
test = pd.DataFrame(test)

#seperating dependent and independent variable 
x = train.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,13]].values
y = train.iloc[:,12].values

#fitting model predicting test values using rbf
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x,y)

#checking accuracy of SVM classifier on test data
SVM_Accuracy = classifier.score(x,y)
print("Accuracy for SVM Classification is :", SVM_Accuracy)
#the accuracy obtyained using rbf kernel is 65.74%, for achieving more accuracy trying different kernel's

#Fitting SVM model to training dataset using linear kernel 
from sklearn.svm import SVC
classifier=SVC(kernel='linear',gamma='auto')
classifier.fit(x,y)

#checking the accuracy of SVM model on training dataset using linear as kernel
SVM_Accuracy = classifier.score(x,y)
print("Accuracy for SVM Classification is :", SVM_Accuracy)
#the accuracy obtained using linear kernel is better 23% than rbf kernel.

#seperating variables for test data
test=test.iloc[:,1:].values

#predicting grade values on test data
predictions = classifier.predict(test)

#the dataset got converted to array's, rebuilding it to dataframe
test = pd.DataFrame(test)

#creating a seperate dataframe and adding grade coloumn in test dataset
test['Grade'] = predictions

# Assigning the column names as per it's index values
test.columns = ['Area(total)', 'Troom', 'Nbedrooms', 'Nbwashrooms', 'Twashrooms','roof', 'Roof(Area)', 'Lawn(Area)', 'Nfloors', 'API', 'ANB', 'EXPECTED','Grade']

#visualising the results
import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter

fig = fig = category_scatter(x='EXPECTED', y='Area(total)', label_col='Grade', data=test, legend_loc='upper left')
plt.xlabel('Expected Price')
plt.ylabel("Total Area")
plt.title("House Grade Classification using SVM With Respect to Expected Price and Total Area")
