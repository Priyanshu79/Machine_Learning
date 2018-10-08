#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 05:26:27 2017

@author: Rajeev
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the dataset into train and test sets
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()
