#cpu일 때와 cpu일 때의 시간 비교
import time



import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model #load_model : model 을 불러옴
from tensorflow.keras.layers import Dense, Input
import sklearn as sk
print(sk.__version__) #0.24.2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

#1.data
dataset=load_boston()

x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=6666)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train)) #0.0 1.0000000000000002 #부동소수점 연산이기 때문에 단순한 연산오류임. 원래는 1.0나와야하눈디 ! !
print(np.min(x_test), np.max(x_test)) #-0.008298755186722073 1.1478180091225068



# #2. modeling
# model=Sequential()
# model.add(Dense(64, input_dim=13)) # 특성은 항상 많으면 좋음! 데이터가 많으면 좋으니까
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(16))
# model.add(Dropout(0.3))
# model.add(Dense(16))
# model.add(Dropout(0.2))
# model.add(Dense(1))


#2-2.모델구성(함수형)
input1= Input(shape=(13,))
dense1 = Dense(64, name = 'ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64, name = 'ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32, name = 'ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(32, name = 'ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(16, name = 'ys5')(drop4)
drop5 = Dropout(0.3)(dense5)
dense6 = Dense(16, name = 'ys6')(drop5)
drop6 = Dropout(0.3)(dense6)
output1 = Dense(1)(drop6)
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


path = 'C:\\ai5\\_save\\keras32\\k32_01\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_01', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time() #time.time() 현재 시간 반환
hist = model.fit(x_train, y_train, epochs=3000, batch_size=1, verbose = 1, validation_split=0.2, callbacks=[es, mcp]) #hist는 history의 약자,
end_time=time.time() #끝나는 시간 반환

model.save('./_save/keras30_mcp/k32/keras32_3_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 0)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)




print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

# #loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718
# 걸린 시간 :  1.97 초

# # ===================1. save.model 출력 =========================
# loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718
# ===================2. mcp 출력 =========================
# loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718


#dropout============================================
# loss :  26.18733024597168
# R2의 점수 :  0.7587578781411475


#걸린시간
#cpu : 12.79초
#gpu : 59.15초