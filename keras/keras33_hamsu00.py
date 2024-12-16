import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. data
# x=np.array([[1,2,3,4,5],
#            [6,7,8,9,10]])
# y=np.array([1,2,3,4,5]) 행렬 형태 안맞음

x=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
            [9,8,7,6,5,4,3,2,1,0]])
y=np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape) #(3,10)
# print(y.shape) #(10, )

x=x.T
# print(x.shape) #(10,3)


#2. moeling (순차형)
#모델을 만들때 순차적으로 만들건지, 함수로 만들건지 결정해야함. 표현 방식만 다른것. 성능 똑같음.
# model=Sequential()
# model.add(Dense(10, input_shape=(3, )))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))
# model.summary()


# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 10)                40

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================


# #2-2. 모델구성(함수형)
input1 = Input(shape=(3, ))
dense1 = Dense(10, name='ys1')(input1) #name='ys1' 작성하면 레이어 이름 변경됨
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1) #순차형과 달리 가장 마지막에 모델 선언
model.summary()

_________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 3)]               0

#  dense (Dense)               (None, 10)                40

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================


# #3. compile, traning
# model.compile(loss= 'mse', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=1)


# #4. predict
# loss=model.evaluate(x, y)
# results=model.predict([[10,1.3,0]]) #predict의 경우 x의 shape대로 행렬의 형태로 작성해야함.
# print('로스 : ', loss)
# print('[10,1.3,0]의 예측값 : ', results)

