from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import time

#1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape, y.shape) #(442, 10) (442, )


#[실습]
#R2 0.62 이상
x=x.reshape(442,10,1)
x=x/255.
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # scaler=StandardScaler()
# scaler=MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)


#2. modeling
from keras.layers import Dropout
model=Sequential()
model.add(Conv1D(64, (3), input_shape=(10, 1), padding='same')) 
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
model.add(Dense(1, activation='linear'))


# #3. compile
model.compile(loss='mse', optimizer='adam',  metrics=['accuracy', 'acc', 'mse'])

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


path = 'C:\\ai5\\_save\\keras60\\k60_03\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k60_03_', date, '_' , filename])
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

# model.save('./_save/keras39/k39_03/keras39_3_mcp.hdf5')

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

# print("R2의 점수 : ", r2)

# #loss :  2887.905517578125
# # R2의 점수 :  0.6110683426680202

# # loss :  3161.851318359375
# # R2의 점수 :  0.5741744184561032



#cnn
# loss :  6818.28173828125
# ACC :  0.0
# 걸린 시간 :  10.79 초

#CNN1D
# loss :  3559.2353515625
# ACC :  0.0
# 걸린 시간 :  24.51 초
# R2의 점수 :  0.5206563242621793