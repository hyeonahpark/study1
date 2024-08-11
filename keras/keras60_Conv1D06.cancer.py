import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping



#1. data
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data #(569, 30)
y = datasets.target #(569, )
print(x.shape, y.shape) 
print(type(x)) # <class 'numpy.ndarray'> 

# numpy, pandas에서 y의 라벨 종류를 찾아낼 수 있음
# numpy로 y의 종류와 개수 파악  numpy.unique / pandas.valueCount
# 0과 1의 갯수가 몇개인지 찾기
unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

print("고유한 요소:", unique) #고유한 요소: [0 1]
print("각 요소의 개수:", counts) #각 요소의 개수: [212 357]

print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts)
print(pd.value_counts(y)) #1    357 / 0    212

# x=x.to_numpy()
x=x.reshape(569,30,1)
x=x/255.

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=6666)

print(x_train.shape, y_train.shape) # (398, 30) (398, )
print(x_test.shape, y_test.shape) # (171, 30) (171, )

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=StandardScaler()
# # scaler=MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)


#2. modeling
from keras.layers import Dropout
model=Sequential()
model.add(Conv1D(64, (3), input_shape=(30, 1), padding='same')) 
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
model.add(Dense(1, activation='sigmoid'))


#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
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


path = 'C:\\ai5\\_save\\keras60\\k60_07\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k60_07_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras39/k39_07/keras39_07_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 1)
# print("loss : ", loss[0])
# print("ACC : ", round(loss[1], 3))

y_pred = model.predict(x_test) 
y_pred = np.round(y_pred)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_pred)

from sklearn.metrics import r2_score, accuracy_score
# accuracy_score = accuracy_score(y_test, y_pred)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("acc_score : ", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")


#acc_score :  0.9824561403508771
# 걸린 시간 :  1.25 초

#dropout
#acc_score :  0.9883040935672515
# 걸린 시간 :  1.8 초

#cnn
# loss :  0.16625624895095825
# ACC :  0.947
# 걸린 시간 :  5.62 초

#cnn1d
# loss :  0.16582903265953064
# ACC :  0.936
# 걸린 시간 :  5.16 초