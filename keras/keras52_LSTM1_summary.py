import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#RNN은 바로 Dense와 연결 가능. 들어갈 때 3차원, 나올 때 2차원

#1. data
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# print(datasets.shape) #(10, )
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],]
             )

# for i in range(len(datasets) - 3):
#     x = (datasets[i:i+3])
#     y =(datasets[i+3]) 
#     print(x)

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)

# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0],x.shape[1], 1)
print(x.shape) # (7, 3, 1)


# 3-D tensor with shape (batch_size, timesteps, features.)

#2.modeling
model=Sequential()
# model.add(SimpleRNN(10, input_shape=(3,1))) #데이터의 갯수 7을 빼고 나머지를 shape에 넣어줌, #3 : timesteps, features
model.add(LSTM(10, input_shape=(3,1))) #LSTM 사용
# model.add(GRU(10, input_shape=(3,1))) #GRU 사용
model.add(Dense(7))
model.add(Dense(1))
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 565
# Trainable params: 565
# Non-trainable params: 0
# _________________________________________________________________

