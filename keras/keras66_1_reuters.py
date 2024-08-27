from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 10000,
    #maxlen = 100,
    test_split = 0.2,
    )

# print(x_train)
print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)
print(y_train) #[ 3  4  3 ... 25  3 25]
print(np.unique(y_train)) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(len(np.unique(y_train))) # 46

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0]), len(x_train[1])) #87 56
#list 데이터를 numpy 데이터로 바꿔줘야 함.

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) #2376
print("뉴스기사의 최대길이 : ", min(len(i) for i in x_train)) #13
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #145.5398574927633
 

#전처리
x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                        truncating='pre')

print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

# y 원핫
from tensorflow.keras.utils import to_categorical
y_train= to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(8982, 46) (2246, 46)

#2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(10000,100)) # 잘 돌아감
model.add(LSTM(512)) # (None, 10)
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(46, activation='softmax'))
model.summary()

# #3.compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras66\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k66_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)

model.fit(x_train, y_train, epochs=500, batch_size = 32, validation_split=0.2)

#4. predict
result = model.evaluate(x_test, y_test)
print('loss : ', result)
#loss : [0.12683972716331482, 0.6736420392990112]

