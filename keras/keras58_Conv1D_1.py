import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Conv1D, Flatten

#RNN은 바로 Dense와 연결 가능. 들어갈 때 3차원, 나올 때 2차원
#DNN 2차원, CNN 4차원, 

#1. data
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# print(datasets.shape) #(10, )
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]] 
             )

# for i in range(len(datasets) - 3):
#     x = (datasets[i:i+3])
#     y =(datasets[i+3]) 
#     print(x)

y = np.array([4,5,6,7,8,9,10])

#2.modeling
model=Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1))) #데이터의 갯수 7을 빼고 나머지를 shape에 넣어줌, #3 : timesteps, features
model.add(Conv1D(10, 2)) #데이터의 갯수 7을 빼고 나머지를 shape에 넣어줌, #3 : timesteps, features
# model.add(LSTM(15, input_shape=(3,1))) #LSTM 사용
# model.add(GRU(10, input_shape=(3,1))) #GRU 사용
model.add(Flatten())
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(1))
model.summary()

#3.compile
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

#4.predict
results=model.evaluate(x,y)
print('loss:', results)

x_pred=np.array([8,9,10]).reshape(1,3,1) #(1,3,1)
y_pred = model.predict(x_pred)
print('[8,9,10]의 결과 : ', y_pred) 

#loss: 1.2343141916545952e-12
# [8,9,10]의 결과 :  [[11.000002]]