#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time

path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape) # (200000, 201)
print(test_csv.shape) # (200000, 200)
print(sample_submission.shape) # (200000, 1)

# print(train_csv.columns)
#Index(['target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6',
    #    'var_7', 'var_8',
    #    ...
    #    'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
    #    'var_196', 'var_197', 'var_198', 'var_199'],
    #   dtype='object', length=201)

# ################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

# print(test_csv.info())

# print(train_csv.describe())

# ################# x와 y 분리 ######################
x=train_csv.drop(['target'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
# print(x)
# print(x.shape) #(200000, 200)
y=train_csv['target']
# print(y.shape) # (200000,)

unique,counts=np.unique(y, return_counts=True)
print(np.unique(y, return_counts=True)) #이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1], dtype=int64), array([179902,  20098], dtype=int64))

# print("고유한 요소:", unique) #고유한 요소: [0 1]
# print("각 요소의 개수:", counts) #각 요소의 개수: [179902  20098]

# print(pd.Series(y).value_counts)
# print(pd.value_counts(y)) 


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1186)

# print(x_train.shape, y_train.shape) # (180000, 200) (180000,)
# print(x_test.shape, y_test.shape) # (20000, 200) (20000,)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


# # #2. modeling
from keras.layers import Dropout
# model=Sequential()
# model.add(Dense(400, activation='relu', input_dim=200))
# model.add(Dropout(0.3))
# model.add(Dense(600, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(800, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(800, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(600, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid')) #최종 아웃풋 노드는 0과 1이 나와야 함. activation(한정함수, 활성화함수)를 사용하여 값을 0~1사이로 한정시킴 



#2-2.모델구성(함수형)
input1= Input(shape=(200,))
dense1 = Dense(400, name = 'ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(600, name = 'ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(800, name = 'ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(1000, name = 'ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(800, name = 'ys5')(drop4)
drop5 = Dropout(0.3)(dense5)
dense6 = Dense(600, name = 'ys6')(drop5)
drop6 = Dropout(0.3)(dense6)
dense7 = Dense(400, name = 'ys7')(drop6)
drop7 = Dropout(0.3)(dense7)
dense8 = Dense(200, name = 'ys8')(drop7)
drop8 = Dropout(0.3)(dense8)
dense9 = Dense(100, name = 'ys9')(drop8)
drop9 = Dropout(0.3)(dense9)
output1 = Dense(1)(drop9)
model = Model(inputs=input1, outputs = output1)
model.summary()



#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
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


path = 'C:\\ai5\\_save\\keras32\\k32_12\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k32_12_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=50, batch_size=2000, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

model.save('./_save/keras32/k32_12/keras32_12_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 1)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))

# y_pred = model.predict(x_test) 
result = model.predict(test_csv)
result = np.round(result)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_pred)

from sklearn.metrics import r2_score, accuracy_score
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

sample_submission['target'] = result
# sample_submission.to_csv(path + "submission_0724_6.csv")


# loss :  0.23717275261878967
# ACC :  0.912

#dropout
# loss :  0.24040205776691437
# ACC :  0.911
