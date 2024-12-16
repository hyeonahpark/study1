import numpy as np
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time

#1. data
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(20640, 8) (20640, )

#[실습]
#R2 0.59 이상

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=3, )


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler=RobustScaler()
# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. modeling
from keras.layers import Dropout

# model=Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dropout(0.3))
# model.add(Dense(10))
# model.add(Dropout(0.3))
# model.add(Dense(10))
# model.add(Dropout(0.3))
# model.add(Dense(10))
# model.add(Dropout(0.3))
# model.add(Dense(10))
# model.add(Dropout(0.3))
# model.add(Dense(1))

#2-2.모델구성(함수형)
input1= Input(shape=(8,))
dense1 = Dense(10, name = 'ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(10, name = 'ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(10, name = 'ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(10, name = 'ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(10, name = 'ys5')(drop4)
drop5 = Dropout(0.3)(dense5)
output1 = Dense(1)(drop5)
model = Model(inputs=input1, outputs = output1)
model.summary()

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


path = 'C:\\ai5\\_save\\keras32\\k32_02\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_02_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

model.save('./_save/keras32/k32_02/keras32_02_mcp.hdf5')


#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

# loss :  0.49470725655555725
# R2의 점수 :  0.6328572745535846

#dropout
# loss :  0.5061473250389099
# R2의 점수 :  0.6243669509853709