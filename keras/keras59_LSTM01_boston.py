
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model #load_model : model 을 불러옴
from tensorflow.keras.layers import Dense, MaxPool2D, BatchNormalization, Conv1D, Flatten, Conv2D, LSTM
import sklearn as sk
print(sk.__version__) #0.24.2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time
from keras.layers import Dropout

#1.data
dataset=load_boston()

x=dataset.data
y=dataset.target

# print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, random_state=6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=MinMaxScaler()
print(x_train.shape, x_test.shape) #(455, 13, 1) (51, 13, 1)

#2. modeling
model=Sequential()
model.add(LSTM(64, input_shape=(13,1))) 
# model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Dense(16)) 
model.add(Dense(1))

#3. compile
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras59\\k59_01\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k59_01', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time() #time.time() 현재 시간 반환
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose = 1, validation_split=0.2, callbacks=[es, mcp]) #hist는 history의 약자,

end_time=time.time() #끝나는 시간 반환

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 0)
print("loss : ", loss)

y_predict=model.predict(x_test)


from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)
print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

# #loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718
# 걸린 시간 :  1.97 초

# # ===================1. save.model 출력 =========================
# loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718
# ===================2. mcp 출력 =========================
# loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718


#dropout============================================
# loss :  26.18733024597168
# R2의 점수 :  0.7587578781411475


#lstm================================
# loss : 18.05930519104004
# R2의 점수 :  0.8548295733084029
# 걸린 시간 :  35.81 초







