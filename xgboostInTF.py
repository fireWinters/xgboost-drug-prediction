'''
Author: callus
Date: 2023-07-03 13:52:27
LastEditors: callus
Description: tensorflow中使用xgboost，创建一个时间序列预测模型
FilePath: /tensorflow-xgboost-st-drugInventory/xgboostInTF.py
'''
# 在tensorflow中使用xgboost，创建一个时间序列预测模型
import tensorflow as tf
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    xgb.XGBClassifier()
])
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
# 加载数据
boston = load_boston()  
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)
# 评估模型
model.evaluate(X_test,  y_test, verbose=2)
# 预测
predictions = model.predict(X_test)
print('predictions:',predictions)
# 保存模型
model.save('my_model.h5')
# 加载模型
new_model = tf.keras.models.load_model('my_model.h5')
# 显示网络结构
new_model.summary()


