#네이버, 하이브 앙상블 => 성우하이텍 월요일 종가 맞추기

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Bidirectional, Flatten, Dropout, LSTM
import pandas as pd
#1. data

#경로 설정
path_data = "C:\\ai5\\_data\\중간고사데이터\\"
path_save = "C:\\ai5\\_save\\중간고사가중치\\"

#데이터 불러오기, 인덱스x
x1_datasets = pd.read_csv(path_data + "naver.csv", index_col=0, thousands=',') #네이버 종가   
x2_datasets = pd.read_csv(path_data + "hybe.csv", index_col=0, thousands=',')  #하이브 종가                                   
y_datasets = pd.read_csv(path_data + "swhitech.csv", index_col=0, thousands=',') #성우하이텍

x1_datasets.columns= ['시가','고가','저가','종가','전일비2','전일비','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']
x2_datasets.columns= ['시가','고가','저가','종가','전일비2','전일비','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']

#하이브에 맞춰 2020년 10월 15일부터의 데이터만 쓰기
x1_datasets = x1_datasets.head(948)
y_datasets = y_datasets.head(948)

#내림차순으로 되어있는 데이터 오름차순으로 정렬
x1_datasets = x1_datasets.sort_values(by=['일자'])
x2_datasets = x2_datasets.sort_values(by=['일자'])
y_datasets = y_datasets.sort_values(by=['일자'])

#필요없는 열 날리기
x1_datasets = x1_datasets.drop(['전일비2','전일비','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'], axis=1)
x2_datasets = x2_datasets.drop(['전일비2','전일비','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'], axis=1)
y = y_datasets['종가']
# print(x1_datasets.shape, x2_datasets.shape, y.shape) # (949, 7) (949, 7) (949,)
# print(y_datasets.isna().sum())
# print(x2_datasets.info())

#8월1일~8월16일 데이터가 x_pred
x1_pred = x1_datasets.tail(11) # 예측해야하는 11개의 데이터 
x2_pred = x1_datasets.tail(11) # 예측해야하는 11개의 데이터 
x1_pred = x1_pred.to_numpy()
x2_pred = x2_pred.to_numpy()

x1_pred = x1_pred.reshape(1,11,7)
x2_pred = x2_pred.reshape(1,11,7)

#reshape을 위해 pandas 데이터 -> numpy 데이터로 변환
x1_datasets = x1_datasets.to_numpy()
x2_datasets = x2_datasets.to_numpy()
y = y.to_numpy()


#split
size = 11
def split_x(dataset, size) :
    aaa=[]
    for i in range(len(dataset)-size+1):
        subset = dataset[i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(x1_datasets, size)
x2 = split_x(x2_datasets, size)
y = split_x(y, size)

print(x1.shape, x2.shape, y.shape) #(938, 11, 7) (938, 11, 7) (938, 11)

x1= np.delete(x1, -1 , axis = 0)
x2= np.delete(x2, -1 , axis = 0)
y= np.delete(y, 0 , axis = 0)

x1_train, x1_test, x2_train, x2_test, y_train, y_test =train_test_split(
    x1, x2, y, train_size=0.9, random_state=5656
)
print(x1_train.shape, x2_train.shape, y_train.shape) # (843, 11, 7) (843, 11, 7) (843, 11)

# model = load_model('_save/중간고사가중치/k630816_1544_0362-391606.5000.hdf5')

# 2-1. model
input1 = Input(shape=(11,7))
dense1 = Bidirectional(LSTM(128, return_sequences=True))(input1)
dense2 = Bidirectional(LSTM(256, return_sequences=True))(dense1)
dense3 = LSTM(512)(dense2)
dense4 = Dense(256, activation='relu')(dense3)
dense5 = Dense(128, activation='relu')(dense4)
dense6 = Dense(64, activation='relu')(dense5)
output1 = Dense(32, activation='relu')(dense6)
model1 = Model(inputs=input1, outputs = output1)

#2-2. model
input11 = Input(shape=(11,7))
dense11 = Bidirectional(LSTM(128, return_sequences=True))(input11)
dense21 = Bidirectional(LSTM(256, return_sequences=True))(dense11)
dense31 = LSTM(512)(dense21)
dense41 = Dense(256, activation='relu')(dense31)
dense51 = Dense(128, activation='relu')(dense41)
dense61 = Dense(64, activation='relu')(dense41)
output2 = Dense(32, activation='relu')(dense61)
model2 = Model(inputs=input11, outputs = output2)

#2-3. 합체!!
from keras.layers.merge import Concatenate, concatenate
merge1 = Concatenate()([output1, output2])
# merge1 = concatenate([output1, output11])
# merge2 = Dense(7, name='mg2')(merge1)
# merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(11, name='last')(merge1)
model = Model(inputs=[input1, input11], outputs = last_output)



#3. compile
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
import time
start_time = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
# print(date) #2024-07-26 16:49:57.565880
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
# print(date) #0726_1654
# print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\중간고사가중치\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k63', date, '_' , filename])
# #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# ################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)

model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=4, validation_split=0.2, callbacks=[es, mcp])
end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# #4. predict

loss=model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_pred, x2_pred])

# print(y_predict)
# print(y_predict.shape)

print("성우하이텍 19일 종가 예측 : ", y_predict)
print("loss : ", loss[0])

# y_predict = pd.DataFrame(y_predict)

# y_predict.to_csv(path + "종가_박현아1.csv", index = False)
