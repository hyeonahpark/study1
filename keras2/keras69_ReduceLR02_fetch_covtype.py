from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)

#1. data

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

y=pd.get_dummies(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y) #stratify : 정확하게 train_size 비율대로 잘라줌

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


# 2. modeling
from keras.layers import Dropout
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim=54))
model.add(Dense(200,  activation = 'relu'))
model.add(Dense(300,  activation = 'relu'))
model.add(Dense(400,  activation = 'relu'))
model.add(Dense(300,  activation = 'relu'))
model.add(Dense(200,  activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(7, activation='softmax'))


#3. compile
from tensorflow.keras.optimizers import Adam
learning_rate = 0.005 #default : 0.001
model.compile(loss='categorical_crossentropy', optimizer = Adam(learning_rate=learning_rate), metrics=['accuracy','acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=1, factor=0.8) #factor는 곱하기! 
################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path_save = 'C:\\ai5\\_save\\keras69\\k69_02\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path_save, 'k69_02_', date, '_' , filename])

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 0,
    save_best_only=True,
    filepath=filepath)

start_time=time.time()
hist=model.fit(x_train, y_train, epochs=50, batch_size=1024, verbose=0, validation_split=0.2, callbacks=[es, mcp])
end_time=time.time()


#4.predict
loss=model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

from sklearn.metrics import r2_score, accuracy_score
print("결과.lr :", learning_rate)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 6))
print("걸린 시간 : ", round(end_time - start_time, 2), "초")


# loss :  0.18621324002742767
# ACC :  0.945

#dropout
#loss :  0.42781001329421997
# ACC :  0.826

#걸린시간
#cpu : 119.31초
#gpu : 53.51초


# 결과.lr : 0.005
# loss :  0.15668454766273499
# ACC :  0.938315
# 걸린 시간 :  71.64 초   
