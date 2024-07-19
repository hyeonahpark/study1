
#https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. data

path = "./_data/따릉이/"

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
print(x) # [1328 rows x 9 columns]

y = train_csv['count'] #train_csv에서 count 열만 y에 넣기
print(y.shape) #(1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

#2. modeling
model=Sequential()
model.add(Dense(333, input_dim=9))
model.add(Dense(100))
model.add(Dense(33))
model.add(Dense(3))
# model.add(Dense(3))
model.add(Dense(1))

#3. compile

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=10,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


model.compile(loss='mse', optimizer='adam')
start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.3, callbacks=[es])
end_time=time.time()

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) #(715, 1)

#loss :  2647.668212890625, R2의 점수 :  0.6120315107071421
#loss :  2872.68505859375, R2의 점수 :  0.6217674592277587
#loss :  2852.4892578125, R2의 점수 :  0.6244266067259094
#loss :  2551.23095703125, R2의 점수 :  0.6290190769131587
#loss :  2500.645751953125, R2의 점수 :  0.6363747477969162 #9 33 11 11 9 3 1, batch_size = 5, random_state=52465
#loss :  2505.2890625, R2의 점수 :  0.6356995739864333 #9 33 11 11 9 1
#loss :  2493.705322265625, R2의 점수 :  0.6373839744298682 # 33 10 33 10 3 1, batch_size=32, random_state=52465
#loss :  2212.40380859375, R2의 점수 :  0.5736656791952154 #33 10 33 10 3 1, batch_size=32, random_state=9999
#loss :  2384.79296875, R2의 점수 :  0.5404460018283344 #33 10 33 22 10 3 1
#loss :  2391.62109375, R2의 점수 :  0.5391302301689438 #333 100 33 22 100 30 1
#loss :  1700.854736328125, R2의 점수 :  0.6936561707539842 #333 100 33 3 1, batch_size=32, random_state=4343
#loss :  1692.931884765625 #333 111 222 111 33 1, batch_size=16, 
#loss :  1688.767822265625, R2의 점수 :  0.6958331807431121 #333 111 222 111 1

#############  submission.csv 만들기 // count 컬럼에 값 넣어주기

submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape) # (715, 1)

# submission_csv.to_csv(path + "submission_0716_5.csv")
# print("loss : ", loss)
# print("R2의 점수 : ", r2)

#갱신
#loss :  1882.2001953125, R2의 점수 :  0.6609937221181643
#loss :  1785.688720703125, R2의 점수 :  0.6783765737955125

print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

print("=================================hist======================================")
print(hist) 
print("==============================hist.history======================================")
print(hist.history) #loss와 val_loss가 epochs 수 만큼 출력됨
print("==============================loss======================================")
print(hist.history['loss']) #history에서 loss 값만 따로 출력
print("==============================val_loss======================================")
print(hist.history['val_loss']) #history에서 val_loss 값만 따로 출력


import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) #그림판 사이즈 설정
plt.plot(hist.history['loss'], c='red', label='loss',)
plt.plot(hist.history['val_loss'], c='blue', label='val_loss',)
plt.legend(loc='upper right') #범례, 범례 위치 위에 오른쪽
plt.title('Ddarung Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

# plt.plot(hist.history['loss'], c='red', label='loss', marker='.')

plt.show()