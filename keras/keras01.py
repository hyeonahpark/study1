import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
print(tf.__version__)

#1. data
x=np.array([1,2,3])
y=np.array([1,2,3])

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile, traning
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. result, predict
result=model.predict([4])
print("4의 예측값 :", result)