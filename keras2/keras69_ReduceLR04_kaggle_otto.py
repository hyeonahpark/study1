from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

#https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from sklearn.model_selection import train_test_split


path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path +"sampleSubmission.csv", index_col=0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])

x=train_csv.drop(['target'], axis=1)
y=train_csv['target']

unique,counts=np.unique(y, return_counts=True)
y=pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. modelinig
from keras.layers import Dropout
model=Sequential()
model.add(Dense(1024, input_dim=93, activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(9, activation='softmax'))


#3. compile
from tensorflow.keras.optimizers import Adam
learning_rate = 0.005
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=0, factor=0.8) #factor는 곱하기!
################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")



path_save = 'C:\\ai5\\_save\\keras69\\k69_04\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path_save, 'k69_04_', date, '_' , filename])


mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 0,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=0, validation_split=0.3, callbacks=[es, mcp, rlr])
end_time=time.time()

#4. predict
loss=model.evaluate(x_test, y_test, verbose = 0)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# y_submit = model.predict(test_csv) # type: ignore

from sklearn.metrics import r2_score, accuracy_score
print("#####################################")
print("결과.lr :", learning_rate)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 6))
print("걸린 시간 : ", round(end_time - start_time, 2), "초")

# loss :  0.6233358979225159
# ACC :  0.783

#걸린시간
#cpu : 87.84초
#gpu : 111.26초

#####################################
# 결과.lr : 0.1
# loss :  1.9552332162857056
# ACC :  0.262767
# 걸린 시간 :  96.95 초
# #####################################
# 결과.lr : 0.01
# loss :  1.281468152999878
# ACC :  0.480284
# 걸린 시간 :  44.03 초
# #####################################
# 결과.lr : 0.005
# loss :  0.8921533823013306
# ACC :  0.709922
# 걸린 시간 :  114.88 초
# #####################################
# 결과.lr : 0.001
# loss :  0.6618871688842773
# ACC :  0.744021
# 걸린 시간 :  101.48 초
# #####################################
# 결과.lr : 0.0005
# loss :  0.6425262689590454
# ACC :  0.758888
# 걸린 시간 :  107.7 초
# #####################################
# 결과.lr : 0.0001
# loss :  0.6283178925514221
# ACC :  0.779573
# 걸린 시간 :  174.85 초


######################################
# 결과.lr : 0.005
# loss :  0.7602882385253906
# ACC :  0.723659
# 걸린 시간 :  236.78 초