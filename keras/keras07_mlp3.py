from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x=np.array(range(10)) #range 함수 / 시작 숫자는 항상 0
print(x) #[0 1 2 3 4 5 6 7 8 9]
print(x.shape) #(10,)

x=np.array(range(1,10)) #1부터 시작하는 숫자를 부를 때 (1, ) 입력, range(n)인경우 0부터 n-1까지 나옴
print(x)

x=np.array(range(1,11))
print(x)
print(x.shape)

x=np.array([range(10), range(21,31), range(201,211)])
print(x)
print(x.shape)
x=x.T
print(x)
print(x.shape)

y=np.array([1,2,3,4,5,6,7,8,9,10])

#[실습]
#[10, 31, 211] 예측할 것

#2. modeling
model=Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. compile, traning
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4.predict
loss=model.evaluate(x,y)
results=model.predict([[10,31,211]])
print('로스 : ',loss)
print('[10, 31, 211]의 예측값 : ', results)