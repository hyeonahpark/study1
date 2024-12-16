from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time

#1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442, )


#[실습]
#R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=StandardScaler()
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. modeling
from keras.layers import Dropout
# model=Sequential()
# model.add(Dense(251, input_dim=10))
# model.add(Dropout(0.3))
# model.add(Dense(141))
# model.add(Dropout(0.3))
# model.add(Dense(171))
# model.add(Dropout(0.3))
# model.add(Dense(14))
# model.add(Dropout(0.3))
# model.add(Dense(5))
# model.add(Dropout(0.3))
# model.add(Dense(1))

#2-2.모델구성(함수형)
input1= Input(shape=(10,))
dense1 = Dense(251, name = 'ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(141, name = 'ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(171, name = 'ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(14, name = 'ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(5, name = 'ys5')(drop4)
drop5 = Dropout(0.3)(dense5)
output1 = Dense(1)(drop5)
model = Model(inputs=input1, outputs = output1)
model.summary()


# #3. compile
model.compile(loss='mse', optimizer='adam')

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


path = 'C:\\ai5\\_save\\keras32\\k32_03\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_03_', date, '_' , filename])
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

model.save('./_save/keras32/k32_03/keras32_3_mcp.hdf5')

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)
print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

#loss :  2887.905517578125
# R2의 점수 :  0.6110683426680202

# loss :  3161.851318359375
# R2의 점수 :  0.5741744184561032



#걸린시간
#cpu :  1.38초
#gpu :  4.78초
