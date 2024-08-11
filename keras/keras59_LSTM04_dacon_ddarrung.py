#https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. data

path = "C:\\Users\\guskek\\ai5\\_data\dacon\\따릉이\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(submission_csv) # [715 rows x 1 columns]

print(train_csv.shape) #(1459,10)
print(test_csv.shape) #(715,10)
print(submission_csv.shape) #(715,1)

print(train_csv.columns) # 컬럼명 출력 (['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #   dtype='object')

# print(train_csv.info())

######################결측치 처리 1. 삭제 #############################

# print(train_csv.isnull().sum())
print(train_csv.isna().sum())
train_csv=train_csv.dropna() #결측치 포함 행 제거
print(train_csv.isna().sum())
print(train_csv)

print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean()) #fillna 함수 : 결측치를 채운다
print(test_csv.info())

x = train_csv.drop(['count'], axis=1) #train_csv에서 count 열 삭제 후 x에 넣기
# print(x.shape) # [1328 rows x 9 columns]

x=x.to_numpy()

y = train_csv['count'] #train_csv에서 count 열만 y에 넣기
# print(y.shape) #(1328,)

x = x.reshape(1328,9,1)
# y = y.reshape(1328,1,1)
x=x/255.

print(x.shape, y.shape) #(1328, 9, 1) (1328,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)


#2. modeling
from keras.layers import Dropout

model=Sequential()
model.add(LSTM(64, input_shape=(9, 1), return_sequences=True))
model.add(LSTM(64, return_sequences=True)) 
model.add(LSTM(32)) 
# model.add(MaxPool2D())
model.add(Dropout(0.25))
# model.add(Conv2D(32, (3,3),  padding='same')) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Flatten()) 
model.add(Dense(units=32))
# model.add(Dropout(0.5))
model.add(Dense(units=16, input_shape=(32, ))) 
model.add(Dense(1, activation='linear'))


# #3. compile
model.compile(loss='mse', optimizer='adam',  metrics=['accuracy', 'acc', 'mse'])
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


path = 'C:\\ai5\\_save\\keras59\\k59_04\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k59_04_', date, '_' , filename])
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


#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
# y_submit = model.predict(test_csv)

# print(y_submit)
# print(y_submit.shape) #(715, 1)

# submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape) # (715, 1)

# submission_csv.to_csv(path + "submission_0716_9.csv")
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  3226.569580078125
# R2의 점수 :  0.562115010446902


# loss :  3139.39990234375
# R2의 점수 :  0.573944981216884


#LSTM
# loss :  7377.3505859375
# ACC :  0.0
# 걸린 시간 :  5.65 초
# R2의 점수 :  -0.001196798865596982