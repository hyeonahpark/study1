from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. data
x=np.array([range(10), range(21,31), range(201,211)])
y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [10,9,8,7,6,5,4,3,2,1],
            [9,8,7,6,5,4,3,2,1,0]])

print(x.shape) #(3, 10)
print(y.shape) #(3, 10)
x=x.T
y=np.transpose(y)
print(x.shape) #(10, 3)
print(y.shape) #(10, 3)


#2.modeling

#[실습]
# x_predict=[10,31,211]
model=Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

#3. compile, training
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4. predict
loss=model.evaluate(x,y)
results=model.predict([[10,31,211]])
print('로스 : ', loss)
print('[10, 31, 211]의 예측값 : ', results)

