import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Input #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
#1. data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train=x_train.to_numpy()
# x_train = x_train.reshape(60000,28*28)
# x_test = x_test.reshape(10000,28*28)

x_train=x_train/255.
x_test=x_test/255.

x_train = x_train.reshape(60000,28*28,1)
x_test = x_test.reshape(10000,28*28,1)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)


# print(x_train) #다 0이 나오는 이유는 특성을 가진 값은 가운데에 몰려있기 때문
# print(x_train[0])
# print("y_train[0] : ", y_train[0]) #5

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> (60000,28,28,1) 과 동일. 데이터의 값과 순서의 변화가 없기 때문
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

#x를 3차원->4차원 데이터로 만들기 reshape

# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

#y는 OneHot Encoding 해서 60000,10으로 만들어주기

# y_train=pd.get_dummies(y_train)
# y_test=pd.get_dummies(y_test)

# print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape) #(10000, 28, 28, 1) (10000, 10)

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y)

# 이미지 데이터는 255로 나누면 자동으로 minmaxScaler임. 이미지의 최솟값은 0, 최댓값은 255기 때문.

# #### 스케일링 1-1 ######
# x_train = x_train/255.
# x_test = x_test/255.

# print(np.max(x_train), np.min(x_train)) #1.0,  0.0


##### 스케일링 1-2 ######
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
# print(np.max(x_train), np.min(x_train)) #1.0 -1.0

#### 스케일링 2. MinMaxScaler(), StandardScaler() #####
# x_train = x_train.reshape(60000,28*28)
# x_test = x_test.reshape(10000,28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train)) #1.0,  0.0


# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

##oneHot 1-1
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# ##oneHot 1-2
# y_train=pd.get_dummies(y_train)
# y_test=pd.get_dummies(y_test)


# ##oneHot 1-3
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.fit_transform(y_test.reshape(-1,1))

# print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape) #(10000, 28, 28, 1) (10000, 10)



#2. modeling
model=Sequential()
model.add(Conv1D(64, (3), input_shape=(28*28, 1), padding='same')) 
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
model.add(Dense(10, activation='softmax'))
                        
                        
#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras60\\k60_14\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k60_14_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=3000, batch_size=128, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras35/keras35_04_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
print(y_predict)


from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# CPU
# loss :  0.3052709698677063
# ACC :  0.918
# 걸린 시간 :  495.78 초

#GPU
#loss :  0.29991039633750916
# ACC :  0.921
# 걸린 시간 :  57.53 초

#loss :  0.05474101006984711
#ACC :  0.985

#loss :  0.061644457280635834
# ACC :  0.986

#loss :  0.06526347994804382
# ACC :  0.987

# loss :  0.062402043491601944
# ACC :  0.99

#maxpool
# loss :  0.039377953857183456
# ACC :  0.993

#hamsu
# loss :  0.18017597496509552
# ACC :  0.96
# 걸린 시간 :  47.54 초

# loss :  0.05910839885473251
# ACC :  0.984
# 걸린 시간 :  103.99 초


#cnn1d
# loss :  0.284437894821167
# ACC :  0.922
# 걸린 시간 :  82.37 초