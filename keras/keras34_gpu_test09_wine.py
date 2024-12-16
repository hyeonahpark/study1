from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

# print(y)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y)) 
#1    71
#0    59
#2    48

y=pd.get_dummies(y)
print(y.shape) #(178, 3)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, shuffle=True, random_state=6666, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# #2. modeling
from keras.layers import Dropout
# model=Sequential()
# model.add(Dense(13, activation='relu', input_dim=13))
# model.add(Dropout(0.3))
# # model.add(Dense(26, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(39, activation='relu'))
# # model.add(Dense(52, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(65, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(65, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(39, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(39, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(13, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(3, activation='softmax'))


#2-2.모델구성(함수형)
input1= Input(shape=(13,))
dense1 = Dense(13, name = 'ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(39, name = 'ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(65, name = 'ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(65, name = 'ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(39, name = 'ys5')(drop4)
drop5 = Dropout(0.3)(dense5)
dense6 = Dense(39, name = 'ys6')(drop5)
drop6 = Dropout(0.3)(dense6)
dense7 = Dense(13, name = 'ys7')(drop6)
drop7 = Dropout(0.3)(dense7)
dense8 = Dense(16, name = 'ys8')(drop7)
drop8 = Dropout(0.3)(dense8)
output1 = Dense(3, activation='softmax')(drop8)
model = Model(inputs=input1, outputs = output1)
model.summary()



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


path = 'C:\\ai5\\_save\\keras32\\k32_10\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_10_', date, '_' , filename])
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

model.save('./_save/keras32/k32_10/keras32_10_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 1)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))

y_pred = model.predict(x_test) 
y_pred = np.round(y_pred)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
print(y_pred)

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  0.16457915306091309
# ACC :  0.944

# loss :  0.36784160137176514
# ACC :  0.944

#걸린시간
#cpu : 1.94초
#gpu : 3.46초