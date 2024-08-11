#https://dacon.io/competitions/official/236068

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Conv1D
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd


#1. data

path = 'C:\\ai5\\_data\\dacon\\diabetes\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)  # (652,9)
print(test_csv.shape)  # (116,8)
print(sample_submission.shape) #(116,1)

################## 결측치 확인 #####################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

###########################################

print(test_csv.info())

print(train_csv.describe())

################# x와 y 분리 ######################
x=train_csv.drop(['Outcome'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
print(x)
print(x.shape) #(652, 8)
y=train_csv['Outcome']
print(y.shape) # (652, )

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

print("고유한 요소:", unique) #고유한 요소: [0 1]
print("각 요소의 개수:", counts) #각 요소의 개수: [424 228]

x=x.to_numpy()
x=x.reshape(652,8,1)
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler=MaxAbsScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)


#2. modeling
from keras.layers import Dropout
model=Sequential()
model.add(Conv1D(64, (3), input_shape=(8, 1), padding='same')) 
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


path = 'C:\\ai5\\_save\\keras60\\k60_06\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k60_06_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras39/k39_06/keras39_06_mcp.hdf5')


#4. predict
loss=model.evaluate(x_test, y_test, verbose = 1)


y_predict = model.predict(x_test)
y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_predict)



y_submit = model.predict(test_csv)
print(y_submit)
# print(y_submit.shape) #(116, 1)

#############  submission.csv 만들기 // count 컬럼에 값 넣어주기

sample_submission['Outcome'] = y_submit
print(sample_submission) 
print(sample_submission.shape) # (116, 2)from sklearn.metrics import r2_score

sample_submission.to_csv(path + "submission_0813.csv")


from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(y_test, y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("acc_score : ", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")



#dropout
#loss :  0.5292198061943054
# ACC :  0.652


#cnn
# loss :  0.5192640423774719
# ACC :  0.788


#cnn1d
# loss :  0.48689761757850647
# ACC :  0.788
# 걸린 시간 :  6.19 초