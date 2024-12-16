#35-4 copy

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Reshape, MaxPooling2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
#1. data

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> (60000,28,28,1) 과 동일. 데이터의 값과 순서의 변화가 없기 때문
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

#### 스케일링 ######
x_train = x_train/255.
x_test = x_test/255.

# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

##oneHot
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.fit_transform(y_test.reshape(-1,1))

# print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape) #(10000, 28, 28, 1) (10000, 10)



#2. modeling
model = Sequential()
model.add(Dense(28, input_shape = (28,28))) #Dense layer 다차원 가능함, (N, 28, 28)
model.add(Reshape(target_shape=(28,28,1))) #(N, 28, 28, 1)
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))) #26, 26, 64
model.add(MaxPooling2D()) #13, 13, 64
model.add(Conv2D(5, (4,4),)) #10, 10, 5
# model.add(Reshape(target_shape=(10*10,5))) #
model.add(Reshape(target_shape=(10*10*5,))) #500,
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
                        
model.summary()
                        
          
          
"""                
#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras35\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k35_04_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=3000, batch_size=128, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras35/keras35_04_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
print(y_predict)


from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# CPU
# loss :  0.3052709698677063
# ACC :  0.918
# 걸린 시간 :  495.78 초

#GPU
#loss :  0.29991039633750916
# ACC :  0.921
# 걸린 시간 :  57.53 초

#loss :  0.05474101006984711
#ACC :  0.985

"""