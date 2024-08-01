# 35-7 copy

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. data

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True)) 
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    #    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), 
    # array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))
    
# #### 스케일링 1-1 ######
x_train = x_train/255.
x_test = x_test/255.

# ##### 스케일링 1-2 ######
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
# # print(np.max(x_train), np.min(x_train)) #1.0 -1.0

# ### 스케일링 2. MinMaxScaler(), StandardScaler() #####
# x_train = x_train.reshape(50000, 32*32*3)
# x_test = x_test.reshape(10000, 32*32*3)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train)) #1.0,  0.0

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.fit_transform(y_test.reshape(-1,1))

#2. modeling
model=Sequential()
model = Sequential()
model.add(Conv2D(100, (3,3), input_shape=(32,32,3))) #27, 27, 10
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임

model.add(Conv2D(filters=50, kernel_size=(3,3))) # 25, 25, 20
model.add(Conv2D(20, (2,2))) # 22, 22, 15
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0

model.add(Dense(units=100)) #None, 22, 22, 8 #Dense가 2차원이지만 2차원 이상 다 가능함
model.add(Dropout(0.2))
model.add(Dense(units=10,)) #22, 22, 9
                        #shpae = (batch_size, input_dim)
model.add(Dense(100, activation='softmax'))
                        
# model.summary()
# model= load_model('./_save/keras35/k35_07/best/k35_07_0731_1037_0029-1.8593.hdf5')

# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3))) 
#                         #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
#                         #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
# model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3))) 
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  
# model.add(Dense(units=32, activation='relu')) 
# model.add(Dense(units=16, input_shape=(32,), activation='relu')) 
#                         #shpae = (batch_size, input_dim)
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax'))

#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)

# ################## mcp 세이브 파일명 만들기 시작 ###################
# import datetime
# date = datetime.datetime.now()
# print(date) #2024-07-26 16:49:57.565880
# print(type(date)) #<class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")
# print(date) #0726_1654
# print(type(date)) #<class 'str'>

# path = 'C:\\ai5\\_save\\keras35\\k35_07\\'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
# filepath = "".join([path, 'k35_07_', date, '_' , filename])
# #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# ################## mcp 세이브 파일명 만들기 끝 ###################

# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath=filepath)


# start_time=time.time()
hist=model.fit(x_train, y_train, epochs=3000, batch_size=12000, validation_split=0.2, callbacks=[es])
# end_time=time.time()

# model.save('./_save/keras35/keras35_07_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
# print(y_predict)


from sklearn.metrics import r2_score, accuracy_score
# accuracy_score = accuracy_score(y_test, y_predict)
# print("loss : ", loss[0])
# print("ACC : ", round(loss[1], 3))
# # print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  1.8817185163497925
# ACC :  0.512

# loss :  2.9173665046691895
# ACC :  0.32