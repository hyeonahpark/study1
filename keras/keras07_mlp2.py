import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
# x=np.array([[1,2,3,4,5],
#            [6,7,8,9,10]])
# y=np.array([1,2,3,4,5]) 행렬 형태 안맞음

x=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
            [9,8,7,6,5,4,3,2,1,0]])
y=np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape) #(3,10)
# print(y.shape) #(10, )

x=x.T
# print(x.shape) #(10,3)


#2. moeling
model=Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. compile, traning
model.compile(loss= 'mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. predict
loss=model.evaluate(x, y)
results=model.predict([[10,1.3,0]]) #predict의 경우 x의 shape대로 행렬의 형태로 작성해야함.
print('로스 : ', loss)
print('[10,1.3,0]의 예측값 : ', results)

