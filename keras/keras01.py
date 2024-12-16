import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
print(tf.__version__)

#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. modeling
model = Sequential()
model.add(Dense(1, input_dim=1)) #input 한덩어리, output 한덩어리

#3. compile, traning
model.compile(loss='mse', optimizer='adam') #compile
model.fit(x, y, epochs=10)

#4. result, predict
result=model.predict(([4]))
print("4의 예측값: ", result)
