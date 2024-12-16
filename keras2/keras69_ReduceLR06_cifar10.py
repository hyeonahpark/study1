import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
model.add(Dense(10, activation='softmax'))


#3. compile
from tensorflow.keras.optimizers import Adam
learning_rate=0.0005
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=1, factor=0.8) #factor는 곱하기!

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")


path_save = 'C:\\ai5\\_save\\keras69\\k69_06\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path_save, 'k69_06_', date, '_' , filename])


mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=1024, verbose=0, validation_split=0.2, callbacks=[es, mcp, rlr])
end_time=time.time()


#4. predict
loss=model.evaluate(x_test, y_test, verbose=1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)

print("##############################################")
print("결과.lr :", learning_rate)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 6))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  1.4850677251815796
# ACC :  0.479
# 걸린 시간 :  148.31 초


##############################################
# 결과.lr : 0.1
# loss :  2.3026087284088135
# ACC :  0.1
# 걸린 시간 :  6.61 초
# ##############################################
# 결과.lr : 0.01
# loss :  2.2812490463256836
# ACC :  0.1064
# 걸린 시간 :  2.79 초
# ##############################################
# 결과.lr : 0.005
# loss :  1.4803460836410522
# ACC :  0.4837
# 걸린 시간 :  8.83 초
# ##############################################
# 결과.lr : 0.001
# loss :  1.410955786705017
# ACC :  0.5035
# 걸린 시간 :  7.21 초
# ##############################################
# 결과.lr : 0.0005
# loss :  1.3875283002853394
# ACC :  0.5143
# 걸린 시간 :  9.18 초
# ##############################################
# 결과.lr : 0.0001
# loss :  1.4063704013824463
# ACC :  0.5017
# 걸린 시간 :  15.85 초


#############################################
##############################################
# 결과.lr : 0.0005
# loss :  1.4161587953567505
# ACC :  0.5064
# 걸린 시간 :  12.45 초