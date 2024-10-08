from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, LSTM
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#1. data

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# print(datasets)
# print(x.shape, y.shape) # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# print(pd.value_counts(y)) 
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

y=pd.get_dummies(y)
# print(y.shape) #(581012, 7)
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
# y=pd.DataFrame(y)
# y= ohe.fit_transform(y)

# print(y.shape) #(581012, 7)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape) #(581012, 8)
# print(y)

# y = np.delete(y, 0 , axis = 1)
# print(y.shape) #(581012, 7)

# x=x.to_numpy()
x=x.reshape(581012,54,1)
x=x/255.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler=RobustScaler()
# scaler=MinMaxScaler()
# scaler.fit(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=6666, stratify=y) #stratify : 정확하게 train_size 비율대로 잘라줌
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# print(pd.value_counts(y_train))
#2    254881
# 1    190768
# 3     32108
# 7     18477
# 6     15656
# 5      8555
# 4      2465


# 2. modeling
from keras.layers import Dropout
model=Sequential()
model.add(LSTM(64, input_shape=(54, 1), return_sequences=True))
model.add(LSTM(64, return_sequences = True)) 
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=True)) 
# model.add(MaxPool2D())
# model.add(Dropout(0.3))
model.add(LSTM(32)) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  #23*23*32
model.add(Dense(units=32))
model.add(Dropout(0.5))
model.add(Dense(units=16, input_shape=(32, ))) 
                        #shpae = (batch_size, input_dim)
model.add(Dense(7, activation='softmax'))


#3. compile

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])
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


path = 'C:\\ai5\\_save\\keras59\\k59_10\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k59_10_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=1000, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras39/k39_10/keras39_10_mcp.hdf5')

#4.predict

loss=model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
# print(y_predict)

from sklearn.metrics import r2_score, accuracy_score
# accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# loss :  0.18621324002742767
# ACC :  0.945

#dropout
#loss :  0.42781001329421997
# ACC :  0.826


#cnn
# loss :  0.2420801967382431
# ACC :  0.911

#LSTM
# loss :  0.6324830055236816
# ACC :  0.718
# 걸린 시간 :  138.78 초