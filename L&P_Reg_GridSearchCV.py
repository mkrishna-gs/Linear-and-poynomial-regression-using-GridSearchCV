# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 00:06:31 2019

@author: mkrishna
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
### Pipeline for linear and polynomial regressions
def PolynomialRegression(degree=2,**kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))

def make_data(N,err=1.0,rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N,1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X,y

#X,y = make_data(40)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
X = df_train['x']
y = df_train['y']
X_test = df_test['x']
y_test = df_test['y']

X = np.array(X)
y = np.array(y)
X_test = np.array(X_test)
y_test = np.array(y_test)

X = X.reshape(-1,1)
X_test = X_test.reshape(-1,1)
### Intialize the hyper parameters 
param_grid = {'polynomialfeatures__degree':np.arange(21),
              'linearregression__fit_intercept':[True,False],
              'linearregression__normalize':[True,False]}
### Find out the degree of the polynomial which provides the best fit for the given data
grid = GridSearchCV(PolynomialRegression(),param_grid,cv=7)
grid.fit(X,y)
model = grid.best_estimator_
### Prediction using the parameters obtained from GridSearchCV
y_pred = model.fit(X,y).predict(X_test)
print(r2_score(y_test,y_pred))
print(grid.best_params_)






