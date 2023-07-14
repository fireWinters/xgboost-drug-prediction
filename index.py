'''
Author: callus
Date: 2023-07-01 21:44:16
LastEditors: callus
Description: some description
FilePath: /tensorflow-xgboost-st-drugInventory/index.py
'''
import tensorflow as tf;
# 加载并准备 MNIST 数据集。将样本数据从整数转换为浮点数：

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 通过堆叠层来构建 tf.keras.Sequential 模型。

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
predictions
print('predictions:',predictions)
aa=tf.nn.softmax(predictions).numpy()
print('softmax:',aa)
# 使用 losses.SparseCategoricalCrossentropy 为训练定义损失函数，它会接受 logits 向量和 True 索引，并为每个样本返回一个标量损失。
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lossA=loss_fn(y_train[:1], predictions).numpy()
print('loss_fn:',lossA)
model.compile(optimizer='adam', 
                loss=loss_fn,
                metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
# Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上检查模型性能。
model.evaluate(x_test,  y_test, verbose=2)
# 如果您想让模型返回概率，可以封装经过训练的模型，并将 softmax 附加到该模型：
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
print('probability_model',probability_model(x_test[:5]))