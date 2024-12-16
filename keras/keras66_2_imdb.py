from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = 10000,
    #maxlen = 100,
)

# print(x_train)
# print(x_train.shape, x_test.shape) #(25000,) (25000,)
# print(y_train.shape, y_test.shape) #(25000,) (25000,)
# print(y_train) #[1 0 0 ... 0 1 0]
# print(np.unique(y_train)) #[0 1]
# print(len(np.unique(y_train))) # 46

# print(type(x_train)) #<class 'numpy.ndarray'>
# print(type(x_train[0])) #<class 'list'>
# print(len(x_train[0]), len(x_train[1])) #218 189
# #list 데이터를 numpy 데이터로 바꿔줘야 함.

# print("imdb 최대길이 : ", max(len(i) for i in x_train)) #2497
# print("imdb 최대길이 : ", min(len(i) for i in x_train)) #11
# print("imdb 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #238.71364

# 전처리
x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                        truncating='pre')

# print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)
# print(y_train.shape, y_test.shape) #(8982,) (2246,)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# x = x_train + x_test
# y = y_train + y_test

#전처리
# x = pad_sequences(x, padding='pre', maxlen=100,
                        # truncating='pre')


# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU ,Bidirectional

model = Sequential()
model.add(Embedding(10000,100)) # 잘 돌아감
model.add(Bidirectional(LSTM(512, return_sequences=True))) # (None, 10)
model.add(LSTM(256)) # (None, 10)
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# #3.compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=500, batch_size = 256, validation_split=0.2, callbacks=[es])

#4. predict
result = model.evaluate(x_test, y_test)
print('loss : ', result)

#scaling, LSTM
# loss :  [0.6912745833396912, 0.5186799764633179]

#scaling x, LSTM
# loss :  [0.3646175265312195, 0.835319995880127]
#scaling x, LSTM 2번
# loss :  [0.35943546891212463, 0.8427600264549255]