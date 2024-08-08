#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

#y 는 T(degC) 로 잡아라.
#자르는 거 맘대로

#predict해야할 부분
#31.12.2016 00:10:00 ~ 01.01.2017 00:00:00 까지 맞춰라 y 144개
#None , 144
#y의 shape는 (n,144)

#url : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time 
from sklearn.model_selection import train_test_split

path = "C:\\프로그램\\ai5\\_data\\kaggle\\jena\\"

a = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

a = a.head(420407)
b = a.tail(144)
b1 = b.drop(['T (degC)'], axis=1)
x1=a.drop(['T (degC)'], axis=1)

y1=a['T (degC)']

def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)

size = 48

x = split_x(x1, size)
y = split_x(y1, size)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(b1.shape)

"""
(294252, 48, 13)
(294252, 48)    
(126108, 48, 13)
(126108, 48)  
"""

#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(13,1), return_sequences=True))
model.add(LSTM(64,)) 
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Flatten()) 
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    verbose=1,
    patience=1,
    restore_best_weights=True
)

model.fit(x1,y1, epochs=10, batch_size=2048, callbacks=[es])
#4. 평가, 예측
results = model.evaluate(x1,y1)
print('loss : ', results)

x_pred = np.array([b1]).reshape(144,13) #[[[8]]]
print(x_pred.shape)
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님
y_pred = model.predict(x_pred)
print(' 결과', y_pred)
print(y_pred.shape)