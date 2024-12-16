from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time

#1. data

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# print(datasets)
# print(x.shape, y.shape) # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# print(pd.value_counts(y)) 
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

y=pd.get_dummies(y)
# print(y.shape) #(581012, 7)
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
# y=pd.DataFrame(y)
# y= ohe.fit_transform(y)

# print(y.shape) #(581012, 7)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape) #(581012, 8)
# print(y)

# y = np.delete(y, 0 , axis = 1)
# print(y.shape) #(581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=6666, stratify=y) #stratify : 정확하게 train_size 비율대로 잘라줌
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# print(pd.value_counts(y_train))
#2    254881
# 1    190768
# 3     32108
# 7     18477
# 6     15656
# 5      8555
# 4      2465

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler=RobustScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# 2. modeling
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])
model=load_model('./_save/keras30_mcp/k30_10/keras30_10_mcp.hdf5')


#4.predict

loss=model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
print(y_predict)

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# loss :  0.18621324002742767
# ACC :  0.945

# loss :  0.18621324002742767
# ACC :  0.945