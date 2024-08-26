#https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. data

path = "C:\\ai5\\_data\\dacon\\따릉이\\"

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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler=RobustScaler()
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_csv=scaler.transform(test_csv)

from sklearn.decomposition import PCA
pca = PCA(n_components = 9)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
evr = pca.explained_variance_ratio_ # 설명가능한 변화율
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

#2. modeling
from keras.layers import Dropout
for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  
    model=Sequential()
    model.add(Dense(33, input_dim=n[i]))
    model.add(Dropout(0.3))
    model.add(Dense(66))
    model.add(Dropout(0.3))
    model.add(Dense(99))
    model.add(Dropout(0.3))
    model.add(Dense(66))
    model.add(Dropout(0.3))
    model.add(Dense(33))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.add(Dense(1))


    #3. compile
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

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


    path_save = 'C:\\ai5\\_save\\m05\\m05_04\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_04', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 1,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()

    #4. predict
    loss=model.evaluate(x_test1, y_test)
    y_predict = model.predict(x_test1)

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_predict)
    # y_submit = model.predict(test_csv)
    # submission_csv['count'] = y_submit
    
    from sklearn.metrics import r2_score
    r2=r2_score(y_test, y_predict)
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")
    print("R2의 점수 : ", r2)



#loss :  3226.569580078125
# R2의 점수 :  0.562115010446902


# loss :  3139.39990234375
# R2의 점수 :  0.573944981216884

#걸린시간
#cpu : 2.59초
#gpu : 2.93초

#################################################################
# 결과.PCA : 7
# loss :  3338.534912109375
# 걸린 시간 :  2.22 초
# R2의 점수 :  0.5469198919430209

# 결과.PCA : 8
# loss :  3263.720458984375
# 걸린 시간 :  3.03 초
# R2의 점수 :  0.5570732057071167
# 2024-08-22 13:10:05.876925

# 결과.PCA : 9
# loss :  3139.19677734375
# 걸린 시간 :  2.13 초
# R2의 점수 :  0.5739725647935553

# 결과.PCA : 9
# loss :  3159.266845703125
# 걸린 시간 :  2.78 초
# R2의 점수 :  0.5712487528117152