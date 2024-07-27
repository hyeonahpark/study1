#https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. data

path = 'C:\\ai5\_data\\kaggle\\bike-sharing-demand\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
# path = 'C://ai5//_data//bike-sharing-demand//' #역슬래시로 작성해도 상관없음
# path = 'C:/ai5/_data/bike-sharing-demand/' 

train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission=pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  #(10886, 11)
print(test_csv.shape)  #(6493, 8)
print(sampleSubmission.shape) # (6493, 1)

#casual, registered 는 미등록 사용자와 등록 사용자임. casual+registered 의 수는 count와 동일하므로 두 열을 삭제해도 됨.
print(train_csv.columns) #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    # 'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(train_csv.info()) #null 값 확인하기
print(test_csv.info())

print(train_csv.describe()) #count, mean, std, min, 1/4분위, 중위값, 3/4분위, max값 나옴. 어떤 주어진 값들을 크기의 순서대로 정렬했을 때 가장 중앙에 위치하는 값

################## 결측치 확인 #####################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

################# x와 y 분리 ######################

x=train_csv.drop(['casual', 'registered', 'count'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
print(x)
print(x.shape) #(10886, 8)

y=train_csv['count']
print(y.shape) # (10886, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=654)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler=RobustScaler()

# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_csv=scaler.transform(test_csv)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=30,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


#2. modeling
from keras.layers import Dropout

model=Sequential()
model.add(Dense(64, input_dim=8, activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

#3. compile
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras32\\k32_05\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_05_', date, '_' , filename])
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

model.save('./_save/keras32/k32_05/keras32_05_mcp.hdf5')

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) 

# sampleSubmission['count'] = y_submit
# print(sampleSubmission)
# print(sampleSubmission.shape) 

# sampleSubmission.to_csv(path + "submission_0725_3.csv")
print("loss : ", loss)
print("R2의 점수 : ", r2)

print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

# loss :  20940.78125
# R2의 점수 :  0.33510549532806966
# 걸린 시간 :  7.49 초

#dropuout
# loss :  27933.568359375
# R2의 점수 :  0.11307610733857121