'''
Author: callus
Date: 2023-07-03 15:48:47
LastEditors: callus
Description: some description
FilePath: /tensorflow-xgboost-st-drugInventory/onlyXgboost1.py
'''

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('shortage_report_export2017-short.csv')
features = data[['Report ID', 'Drug Identification Number']]   # 请将 'feature1', 'feature2', 'feature3' 替换为你的实际特征名称
labels = data['Tier 3']  # 请将 'label' 替换为你的实际标签名称

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

model = xgb.XGBRegressor()
model.fit(features_train, labels_train)

predictions = model.predict(features_test)
mse = mean_squared_error(labels_test, predictions)
print('MSE: %.2f' % mse)