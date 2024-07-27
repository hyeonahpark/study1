#https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split


path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path +"sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  # (61878, 94)
# print(test_csv.shape)  # (144368, 93)
# print(sampleSubmission_csv.shape) # (144368, 9)

################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

################# x와 y 분리 ######################
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])

x=train_csv.drop(['target'], axis=1)
y=train_csv['target']

# print(x)

# print(x.shape) # (61878, 93)
# print(y.shape) # (61878,)

# print(y)

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

# print("고유한 요소:", unique) #고유한 요소: [0 1 2 3 4 5 6 7 8]
# print("각 요소의 개수:", counts) #각 요소의 개수: [ 1929 16122  8004  2691  2739 14135  2839  8464  4955]

y=pd.get_dummies(y)
# print(y.shape) #(61878, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=8)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. modelinig
from keras.layers import Dropout
model=Sequential()
model.add(Dense(1024, input_dim=93, activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(9, activation='softmax'))

#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
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


path = 'C:\\ai5\\_save\\keras32\\k32_13\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_13_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

model.save('./_save/keras32/k32_13/keras32_13_mcp.hdf5')

#4. predict
loss=model.evaluate(x_test, y_test, verbose = 1)
y_predict = model.predict(x_test)

# print(y_predict)

y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
y_submit = model.predict(test_csv) # type: ignore

# print(y_submit)
# print(y_submit.shape) #(144368, 9)

############  submission.csv 만들기 // count 컬럼에 값 넣어주기

# sampleSubmission_csv = y_submit
# print(sampleSubmission_csv) 
# print(sampleSubmission_csv.shape)

sampleSubmission_csv[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit

# sampleSubmission_csv.to_csv(path + "submission_0724_15.csv")

from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(y_test, y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초")


# loss :  0.6233358979225159
# ACC :  0.783