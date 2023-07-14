'''
Author: callus
Date: 2023-07-03 14:31:11
LastEditors: callus
Description: some description
FilePath: /tensorflow-xgboost-st-drugInventory/xgboost1.py
'''
import tensorflow as tf
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# 载入文件
dataframe = pd.read_csv('shortage_report_export2017-short.csv')
features = dataframe.columns
print(features,'特征值')
# 处理数据
# Report ID	Drug Identification Number	Report Type	Brand name	Company Name	Common or Proper name	Ingredients	Strength(s)	Packaging size	Route of administration	Shortage status	Dosage form(s)	ATC Code	ATC description	Anticipated start date	Actual start date	Estimated end date	Actual end date	Reason	Date Created	Date Updated	Tier 3
features = dataframe[['Report ID', 'Drug Identification Number', 'Report Type', 'Brand name',
       'Company Name', 'Common or Proper name', 'Ingredients', 'Strength(s)',
       'Packaging size', 'Route of administration', 'Shortage status',
       'Dosage form(s)', 'ATC Code', 'ATC description',
       'Anticipated start date', 'Actual start date', 'Estimated end date',
       'Actual end date', 'Reason', 'Date Created', 'Date Updated', 'Tier 3']] 
# 请将 'feature1', 'feature2' 等替换为你的实际特征名称
labels = dataframe['Tier 3'] # 请将 'label' 替换为你的实际标签名称
# # 划分数据集为训练集、验证集和测试集，比例为6:2:2
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4, random_state=0)
test_dataset = tf.data.Dataset.from_tensor_slices((features_test.values, labels_test.values))

train_dataset, test_dataset, val_dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values)).shuffle(len(features)).batch(1)

train_dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    xgb.XGBClassifier()
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(train_dataset, epochs=10, validation_data=val_dataset)
model.fit(train_dataset.batch(32), epochs=10)

# 
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc)

