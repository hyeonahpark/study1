#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, LSTM
from sklearn.model_selection import train_test_split
import time

path = "C:\\Users\\guskek\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
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

x=x.to_numpy()
x=x.reshape(200000,50*4,1)
x=x/255.
y=pd.get_dummies(y)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1186)

# print(x_train.shape, y_train.shape) # (180000, 200) (180000,)
# print(x_test.shape, y_test.shape) # (20000, 200) (20000,)


# # #2. modeling
from keras.layers import Dropout
model=Sequential()
model.add(LSTM(64, input_shape=(200, 1),return_sequences=True)) 
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True)) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(LSTM(32)) 
# model.add(MaxPool2D())
model.add(Dropout(0.25))
# model.add(Conv2D(32, (3,3), activation='relu', padding='same')) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  #23*23*32
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, input_shape=(32, ), activation='relu')) 
                        #shpae = (batch_size, input_dim)
model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid'))


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


path = 'C:\\Users\\guskek\\ai5\\save\\keras59\\k59_12\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k59_12_', date, '_' , filename])
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


#4. predict

loss=model.evaluate(x_test, y_test, verbose = 1)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))

# y_pred = model.predict(x_test) 
# result = model.predict(test_csv)
# result = np.round(result)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_pred)

from sklearn.metrics import r2_score, accuracy_score
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

# sample_submission['target'] = result
# sample_submission.to_csv(path + "submission_0724_6.csv")


# loss :  0.23717275261878967
# ACC :  0.912

#dropout
# loss :  0.24040205776691437
# ACC :  0.911

# loss :  0.2420801967382431
# ACC :  0.911
# 걸린 시간 :  96.17 초


#LSTM
#loss :  0.3284958004951477
# ACC :  0.899
# 걸린 시간 :  55.19 초