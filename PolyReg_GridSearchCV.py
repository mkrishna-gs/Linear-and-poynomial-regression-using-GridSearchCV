from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
X_train = df_train['x'].values.reshape(-1,1)
y_train = df_train['y']
X_test = df_test['x'].values.reshape(-1,1)
y_test = df_test['y']

# Polynomial regression
def PolynomialRegression(degree=2,**kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))

# Intialize the hyper parameters 
param_grid = {'polynomialfeatures__degree':np.arange(21),
              'linearregression__fit_intercept':[True,False],
              'linearregression__normalize':[True,False]}

# Find out the degree of the polynomial which provides the best fit for the given data
grid = GridSearchCV(PolynomialRegression(),param_grid,cv=7)
grid.fit(X_train,y_train)
poly_model = grid.best_estimator_
# Prediction using the parameters obtained from GridSearchCV
y_pred = poly_model.fit(X_train,y_train).predict(X_test)
print(f'R2 score using polynomial regression: {r2_score(y_test,y_pred)}')
#print(grid.best_params_)

#plt.scatter(X_test, y_test, label='actual')
#plt.plot(X_test, y_pred, color='orange', label='predicted')
#plt.legend()
#plt.show()