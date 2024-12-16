import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D, LSTM
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train) #다 0이 나오는 이유는 특성을 가진 값은 가운데에 몰려있기 때문
print(x_train[0])
print("y_train[0] : ", y_train[0]) #5

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> (60000,28,28,1) 과 동일. 데이터의 값과 순서의 변화가 없기 때문
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

#x를 3차원->4차원 데이터로 만들기 reshape

x_train = x_train.reshape(60000,28*28,1)
x_test = x_test.reshape(10000,28*28,1)

#y는 OneHot Encoding 해서 60000,10으로 만들어주기

# y_train=pd.get_dummies(y_train)
# y_test=pd.get_dummies(y_test)

# print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape) #(10000, 28, 28, 1) (10000, 10)

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y)

# 이미지 데이터는 255로 나누면 자동으로 minmaxScaler임. 이미지의 최솟값은 0, 최댓값은 255기 때문.

#### 스케일링 1-1 ######
x_train = x_train/255.
x_test = x_test/255.

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

# import matplotlib.pyplot as plt
# plt.imshow(x_train[4], 'gray') #gray : 흑백
# plt.show()


#2. modeling
model = Sequential()
model.add(LSTM(64, input_shape=(28*28, 1), return_sequences=True)) #26, 26, 64
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True)) # 24, 24, 64
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(LSTM(32)) # 24, 24, 64
# model.add(MaxPool2D())
# model.add(Dropout(0.25))
# model.add(Conv2D(32, (3,3), activation='relu', padding='same')) # 23, 23, 32
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  #23*23*32
model.add(Dense(units=32, activation='relu')) #None, 22, 22, 8 #Dense가 2차원이지만 2차원 이상 다 가능함
model.add(Dense(units=16, input_shape=(32, ), activation='relu')) 
                        #shpae = (batch_size, input_dim)
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
                        
# model.summary()
                        
                        
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


path = 'C:\\ai5\\_save\\keras59\\k59_15\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k59_15_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=1028, validation_split=0.3, callbacks=[es, mcp])
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


# loss :  0.2710987627506256
# ACC :  0.91

# loss :  0.29126986861228943
# ACC :  0.914

# loss :  0.24803583323955536
# ACC :  0.922

#lstm
# loss :  0.7938035130500793
# ACC :  0.7
# 걸린 시간 :  764.44 초