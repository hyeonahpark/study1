import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

#1. data

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

#### 스케일링 1-1 ######
x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.transform(y_test.reshape(-1,1))



#2. modeling
model=Sequential()
model.add(Dense(100, input_shape=(32*32*3, )))
model.add(Dense(200,  activation = 'relu'))
model.add(Dense(300,  activation = 'relu'))
model.add(Dense(600,  activation = 'relu'))
model.add(Dense(300,  activation = 'relu'))
model.add(Dense(200,  activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation='softmax'))


#3. compile
from tensorflow.keras.optimizers import Adam
learning_rate=0.0001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=1, factor=0.8) #factor는 곱하기!

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")


path_save = 'C:\\ai5\\_save\\keras69\\k69_07\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path_save, 'k69_07_', date, '_' , filename])


mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 0,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=1028, verbose=0, validation_split=0.2, callbacks=[es, mcp])
end_time=time.time()


#4. predict
loss=model.evaluate(x_test, y_test, verbose=0)
y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)

print("##############################################")
print("결과.rlr :", learning_rate)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 6))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


##############################################
# 결과.lr : 0.1
# loss :  4.6100664138793945
# ACC :  0.01
# 걸린 시간 :  4.94 초
# ##############################################
# 결과.lr : 0.01
# loss :  4.605957984924316
# ACC :  0.01
# 걸린 시간 :  2.38 초
# ##############################################
# 결과.lr : 0.005
# loss :  4.605404853820801
# ACC :  0.01
# 걸린 시간 :  2.42 초
# ##############################################
# 결과.lr : 0.001
# loss :  3.306180000305176
# ACC :  0.2277
# 걸린 시간 :  7.81 초
# ##############################################
# 결과.lr : 0.0005
# loss :  3.2829387187957764
# ACC :  0.2274
# 걸린 시간 :  8.39 초
# ##############################################
# 결과.lr : 0.0001
# loss :  3.2632641792297363
# ACC :  0.235
# 걸린 시간 :  22.87 초



###############################################
# 결과.rlr : 0.0001
# loss :  3.2389590740203857
# ACC :  0.2392
# 걸린 시간 :  38.13 초