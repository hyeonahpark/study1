#https://dacon.io/competitions/official/236068

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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
print(y.shape) # (652., )

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

print("고유한 요소:", unique) #고유한 요소: [0 1]
print("각 요소의 개수:", counts) #각 요소의 개수: [424 228]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# scaler=MinMaxScaler()
scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)




#2. modeling
from keras.layers import Dropout
# model=Sequential()
# model.add(Dense(16, input_dim=8, activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))



#2-2.모델구성(함수형)
input1= Input(shape=(8,))
dense1 = Dense(16, name = 'ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(32, name = 'ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32, name = 'ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(64, name = 'ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(64, name = 'ys5')(drop4)
drop5 = Dropout(0.3)(dense5)
dense6 = Dense(32, name = 'ys6')(drop5)
drop6 = Dropout(0.3)(dense6)
dense7 = Dense(32, name = 'ys7')(drop6)
drop7 = Dropout(0.3)(dense7)
dense8 = Dense(16, name = 'ys8')(drop7)
drop8 = Dropout(0.3)(dense8)
dense9 = Dense(16, name = 'ys9')(drop8)
drop9 = Dropout(0.3)(dense9)
dense10 = Dense(8, name = 'ys10')(drop9)
drop10 = Dropout(0.3)(dense10)
output1 = Dense(1)(drop10)
model = Model(inputs=input1, outputs = output1)
model.summary()


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


path = 'C:\\ai5\\_save\\keras32\\k32_06\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_06_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

model.save('./_save/keras32/k32_06/keras32_06_mcp.hdf5')


#4. predict
loss=model.evaluate(x_test, y_test, verbose = 1)


y_predict = model.predict(x_test)
y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_predict)



y_submit = np.round(model.predict(test_csv))
# print(y_submit)
# print(y_submit.shape) #(116, 1)

#############  submission.csv 만들기 // count 컬럼에 값 넣어주기

sample_submission['Outcome'] = y_submit
# print(sample_submission) 
# print(sample_submission.shape) # (116, 2)from sklearn.metrics import r2_score

# sample_submission.to_csv(path + "submission_0724_2.csv")


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

#걸린시간
#cpu : 1.03초
#gpu : 2.1초