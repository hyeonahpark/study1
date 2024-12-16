from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

# print(y)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y)) 
#1    71
#0    59
#2    48

y=pd.get_dummies(y)
print(y.shape) #(178, 3)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, shuffle=True, random_state=6666, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# #2. modeling
model=Sequential()
model.add(Dense(13, activation='relu', input_dim=13))
# model.add(Dense(26, activation='relu'))
model.add(Dense(39, activation='relu'))
# model.add(Dense(52, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
model=load_model('./_save/keras30_mcp/k30_9/keras30_9_mcp.hdf5')


#4. predict

loss=model.evaluate(x_test, y_test, verbose = 1)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))

y_pred = model.predict(x_test) 
y_pred = np.round(y_pred)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
print(y_pred)

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  0.16457915306091309
# ACC :  0.944

# loss :  0.16457915306091309
# ACC :  0.944