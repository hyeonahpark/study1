import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time
from keras.models import load_model

#1.데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])

#2.modeling
# model=load_model('./_save/keras52/best/k52_020807_1658_0927-0.0126.hdf5')

model=Sequential()
# model.add(SimpleRNN(10, input_shape=(3,1))) #데이터의 갯수 7을 빼고 나머지를 shape에 넣어줌, #3 : timesteps, features
model.add(LSTM(15, input_length=3, input_dim=1, return_sequences=True))
model.add(LSTM(15))
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(1))

#3.compile
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=10000, batch_size=10)

#4.predict
results=model.evaluate(x,y)
print('loss:', results)

x_pred=np.array([50,60,70]).reshape(1,3,1) #(1,3,1)
y_pred = model.predict(x_pred)
print('[50,60,70]의 결과 : ', y_pred) 

from tensorflow.python.keras.models import load_model
path = "C:\\ai5\\_save\\keras52\\"
model.save(path + 'k52_model1.h5')


#loss: 0.0008884485578164458
# [50,60,70]의 결과 :  [[77.90359]]

#lstm 2번
# loss: 0.11316175013780594
# [50,60,70]의 결과 :  [[78.0855]]

# loss: 2.812367347360123e-05
# [50,60,70]의 결과 :  [[76.21746]]