'''
Author: callus
Date: 2023-07-03 16:42:54
LastEditors: callus
Description: some description
FilePath: /tensorflow-xgboost-st-drugInventory/xgboostXGBRegressor.py
'''
import xgboost as xgb
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

x, y = data, target
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

xgbr = xgb.XGBRegressor(verbosity=0) 
print(xgbr)
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1, gamma=0,
#        importance_type='gain', learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=None, subsample=1, verbosity=1)

# Next, we'll fit the model with train data.

xgbr.fit(xtrain, ytrain)
# After training the model, we'll check the model training score.

score = xgbr.score(xtrain, ytrain)  
print("Training score: ", score)
# Training score:  0.9738225090795732 

# We can also apply the cross-validation method to evaluate the training score.

scores = cross_val_score(xgbr, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())
# Mean cross-validataion score: 0.87 

# Or if you want to use the KFlold method in cross-validation it goes as below.

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, xtrain, ytrain, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
# K-fold CV average score: 0.87


ypred = xgbr.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
# MSE: 3.35
print("RMSE: %.2f" % (mse**(1/2.0)))
# RMSE: 1.83 
x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title("Boston test and predicted data")
plt.legend()
plt.show()
