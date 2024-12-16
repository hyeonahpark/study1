#jena를 DNN으로 구성

#X : (42만, 144, 144) => (42만, 144*144)
#Y : (42만, 144)

#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

#y는 T (degC), 자르는건 맘대로  / 2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00에 대한 데이터를 예측하기 (Y=144) train y.shape=(N,144) -> predict (1,144)
#일요일 23:59:59초까지, 평가지표 rmse, csv파일 제출

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Flatten, GRU
import time

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
path_data = "C:\\ai5\\_data\\kaggle\\jena\\"

dataset=pd.read_csv(path_data + "jena_climate_2009_2016.csv", index_col=0)

# print(sample_submission) 
# print(sample_submission.shape) #(420551, 14)

a = dataset.head(420407) # 예측해야하는 144개를 뺀 나머지 훈련시킬 데이터
a2 = a.tail(144) # 예측해야하는 144개의 데이터 (x_pred)
b = dataset.tail(144) #원래 정답
# print(a)
# print(a.shape) #(420407, 14)
# print(b.shape) #(144, 14)

x_pred = a2.drop(['T (degC)'], axis=1)
# print(x_pred)
# print(x_pred.shape) #(144, 13)

x=a.drop(['T (degC)'], axis=1)
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
# print(x.shape) #(420264, 144, 13)


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
# print(type(x))

y= np.delete(y, 0 , axis = 0)
# print("y변환 : ",y[0])
# print(y.shape) #(420263, 144)


print(x.shape, y.shape) #(420263, 144, 13) (420263, 144)

x = x.reshape(420263,144*13) #(420263, 144*13)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=5656)

# model=load_model('./_save/keras55/jena_박현아.hdf5')


#2. 모델 구성
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(144*13, ))) # timesteps , features
model.add(Dense(128, activation='relu')) # timesteps , features
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu')) # timesteps , features
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(144))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time = time.time()
#3. compile
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
# print(date) #2024-07-26 16:49:57.565880
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
# print(date) #0726_1654
# print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras61\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k61_', date, '_' , filename])
# #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# ################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


model.fit(x_train, y_train, epochs=1000, batch_size=512, validation_split=0.2, callbacks=[es, mcp])
end_time = time.time()


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=512)
print('loss :', result[0])

x_pred=np.array([x_pred]).reshape(1,144*13)
# print(x_pred.shape)

y_pred = model.predict(x_pred, batch_size=512)
y_pred = np.array([y_pred]).reshape(144,1)
# print('결과 :', y_pred)
# print(y_pred.shape)
# print(y_pred.shape)

from sklearn.metrics import r2_score, mean_squared_error
y_test = b['T (degC)']

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)
# print("데이터 처리 걸린 시간 :", round(end_time-start_time,2),'초') #146.38초

###csv 파일 만들기 ###
submit = pd.read_csv(path_data + "jena_climate_2009_2016.csv")

submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
# print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

submit['T (degC)'] = y_pred
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

submit.to_csv(path + "jena_0814.csv", index = False)



#RMSE : 1.1558566105864292, 0.23490005731582642, batch 512, unit 128 시작


#DNN
# loss : 0.5058857202529907
# RMSE : 1.7412381400579398

# loss : 0.1354541778564453
# RMSE : 1.4285867440312014

#loss : 0.36048784852027893
# RMSE : 1.292126606818696

# RMSE : 1.2402703505642076
# loss : 0.1753847301006317