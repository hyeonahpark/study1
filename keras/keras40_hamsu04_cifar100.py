import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Input #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
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

INPUT_SHAPE = (32, 32, 3)
KERNEL_SIZE = (3,3)

#2. modeling
model=Sequential()
input1=Input(shape=(32,32,3))
dense1=Conv2D(filters=256, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same')(input1)
batch1=BatchNormalization()(dense1)
maxp1=MaxPool2D()(batch1)
dense2=Conv2D(filters=256, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same')(batch1)
maxp2=MaxPool2D()(dense2)
batch2=BatchNormalization()(maxp2)
drop1=Dropout(0.5)(batch2)

dense3=Conv2D(filters=512, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same')(drop1)
maxp3=MaxPool2D()(dense3)
batch3=BatchNormalization()(maxp3)
dense4=Conv2D(filters=512, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same')(batch3)
maxp4=MaxPool2D()(dense4)
batch4=BatchNormalization()(maxp4)
drop2=Dropout(0.5)(batch4)

dense5=Conv2D(filters=1024, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same')(drop2)
maxp5=MaxPool2D()(dense5)
batch5=BatchNormalization()(maxp5)
dense6=Conv2D(filters=1024, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same')(batch5)
# model.add(MaxPool2D())
batch6=BatchNormalization()(dense6)
drop3=Dropout(0.5)(batch6)

flat1=Flatten()(drop3)
# model.add(Dropout(0.2))
dense7=Dense(1024, activation='relu')(flat1)
drop4=Dropout(0.7)(dense7)
output1=Dense(100, activation='softmax')(drop4)
model = Model(inputs=input1, outputs = output1)   


#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras40\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k40_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)

start_time=time.time()
hist=model.fit(x_train, y_train, epochs=3000, batch_size=512, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras40/keras40_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
# print(y_predict)


from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# loss :  2.9173665046691895
# ACC :  0.32

#loss :  2.165902853012085
# ACC :  0.434
#32 32 64 64 128 128 128

# loss :  2.5809075832366943
# ACC :  0.442

# loss :  2.0881471633911133
# ACC :  0.471

# loss :  2.0324556827545166
# ACC :  0.491

# loss :  2.1437559127807617
# ACC :  0.486

#loss :  1.8817185163497925
# ACC :  0.512


#hamsu
# loss :  1.7685941457748413
# ACC :  0.537

# loss :  1.7509618997573853
# ACC :  0.539

