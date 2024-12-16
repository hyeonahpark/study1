import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x=np.array([[1,2,3,4,5],
           [6,7,8,9,10]])
y=np.array([1,2,3,4,5])
print(x.shape)
#x=x.T   전치행렬 함수
#x=x.transpose()
#x=np.transpose(x)
#x=np.swapaxes(x,0,1)
print(x.shape)
print(y.shape)

# # x=np.array([[1,6], [2,7], [3,8], [4,9], [5,10]])
# # y=np.array([1,2,3,4,5])

# print(x.shape) #(5,2)
# print(y.shape) #(5, )

# #2. moeling
# model=Sequential()
# model.add(Dense(10, input_dim=2))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))

# #3. compile, traning
# model.compile(loss= 'mse', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=1)

# #4. predict
# loss=model.evaluate(x, y)
# results=model.predict([[6,11]]) #predict의 경우 x의 shape대로 행렬의 형태로 작성해야함.
# print('로스 : ', loss)
# print('[6,11]의 예측값 : ', results)


# #[실습] 소수 2째자리까지 맞추기