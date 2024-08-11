from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

# print(y)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y)) 
#1    71
#0    59
#2    48

# x=x.to_numpy()
x=x.reshape(178,13,1)
x=x/255.

# y=pd.get_dummies(y)
# print(y.shape) #(178, 3)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, shuffle=True, random_state=6666, stratify=y)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=StandardScaler()
# # scaler=MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)

# #2. modeling
from keras.layers import Dropout
model=Sequential()
model.add(Conv1D(64, (3), input_shape=(13, 1), padding='same')) 
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
                        #가중치 = 커널사이즈
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=(2), padding='same')) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=(2), padding='same')) 
# model.add(MaxPool2D())
# model.add(Dropout(0.25))
model.add(Conv1D(32, (2),  padding='same')) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  #23*23*32
model.add(Dense(units=32))
model.add(Dense(units=16, input_shape=(32, ))) 
                        #shpae = (batch_size, input_dim)
# model.add(Dropout(0.25))
model.add(Dense(1, activation='softmax'))


#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
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


path = 'C:\\ai5\\_save\\keras60\\k60_10\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k60_10_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras39/k39_10/keras39_10_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 1)
y_pred = model.predict(x_test) 
y_pred = np.round(y_pred)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_pred)

from sklearn.metrics import r2_score, accuracy_score
# accuracy_score = accuracy_score(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# accuracy_score = accuracy_score(y_test, y_pred)
# print("r2 : ", r2)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  0.16457915306091309
# ACC :  0.944

# loss :  0.36784160137176514
# ACC :  0.944

# loss :  0.0
# ACC :  0.389

#cnn1d
# loss :  0.0
# ACC :  0.389
# 걸린 시간 :  2.81 초