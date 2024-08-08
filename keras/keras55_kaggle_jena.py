#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

#y는 T (degC), 자르는건 맘대로  / 2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00에 대한 데이터를 예측하기 (Y=144) train y.shape=(N,144) -> predict (1,144)
#일요일 23:59:59초까지, 평가지표 rmse, csv파일 제출

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
path_submission = "C:\\ai5\\_data\\kaggle\\jena\\"

dataset=pd.read_csv(path_submission + "jena_climate_2009_2016.csv", index_col=0)

# print(sample_submission) 
# print(sample_submission.shape) #(420551, 14)

a = dataset.head(420407) # 예측해야하는 144개를 뺀 나머지 훈련시킬 데이터
b = dataset.tail(144) # 예측해야하는 144개
# print(a)
# print(a.shape) #(420407, 14)
# print(b.shape) #(144, 14)

x_pred = b.drop(['T (degC)'], axis=1)
# print(x_pred)
# print(x_pred.shape) #(144, 13)

x=a.drop(['T (degC)'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
# print(x)
# print(x.shape) #(420407, 13)

y=a['T (degC)']
# print(y)
# print(y.shape) #(420407, )






size = 144
def split_x(dataset, size) :
    aaa=[]
    for i in range(len(dataset)-size+1):
        subset = dataset[i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x, size)
# print(x[-2])
# print(x[-1])

# x=x[:, :-1]
# print(bbb)
print(x.shape) #(420264, 144, 13)


size2 = 144
def split_y(dataset, size2) :
    aaa=[]
    for i in range(len(dataset)-size2+1):
        subset = dataset[i: (i+size2)]
        aaa.append(subset)
    return np.array(aaa)


y = split_y(y, size2)
# print(y.shape) #(420264, 144)
# print("y : ",y[0])
# print("y : ",y[1])

# print(type(y)) #<class 'numpy.ndarray'>

x= np.delete(x, -1 , axis = 0)
# print("x변환 :", x[-1])
# print(x.shape) #(420263, 144, 13)
print(type(x))

y= np.delete(y, 0 , axis = 0)
# print("y변환 : ",y[0])
# print(y.shape) #(420263, 144)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, GRU

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

# #2. 모델 구성
model = Sequential()
model.add(LSTM(units=64, input_shape=(144,13), return_sequences=True)) # timesteps , features
# model.add(LSTM(32, return_sequences=True)) # timesteps , features
model.add(LSTM(64))
# # Flaten 사용하는 방법도 있음 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(144))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
#3. compile
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


path = 'C:\\ai5\\_save\\keras55\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k55_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


model.fit(x_train, y_train, epochs=1000, batch_size=400, validation_split=0.2, callbacks=[es, mcp])


#4. 평가, 예측
result = model.evaluate(x, y, batch_size=400)
print('loss :', result)

x_pred=np.array([x_pred]).reshape(1,144,13)
# print(x_pred.shape)

y_pred = model.predict(x_pred, batch_size=400)
y_pred = np.array([y_pred]).reshape(144,1)
print('결과 :', y_pred)
# print(y_pred.shape)


from sklearn.metrics import r2_score, mean_squared_error
y_test = b['T (degC)']

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)


#RMSE : 1.9420303450211813, batch 2000
#RMSE : 1.443340110519598, batch 1000, loss : 0.6316823363304138, patience 10
#RMSE : 0.6372527017565991, batch 400, loss :0.21237652003765106, patience 30

#standardscaler
#RMSE : 3.349785771425728

#minmaxscaler