import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.applications import VGG16

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

#### 스케일링 1-1 ######
x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
#y_train= ohe.fit_transform(y_train.reshape(-1,1))
#y_test= ohe.fit_transform(y_test.reshape(-1,1))

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0]) #gray : 흑백
# plt.show()
KERNEL_SIZE = (3, 3)
INPUT_SHAPE = (32, 32, 3)


vgg16 = VGG16(#weights='imagenet',
              include_top=False,
              input_shape=(32,32,3)
)
vgg16.trainable = True # false : 가중치 동결


#2. modeling
model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

"""
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3))) 
#                         #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
#                         #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
# model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3))) 
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  
# model.add(Dense(units=32, activation='relu')) 
# model.add(Dense(units=16, input_shape=(32,), activation='relu')) 
#                         #shpae = (batch_size, input_dim)
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax'))
"""

#3. compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>

path = 'C:\\ai5\\_save\\keras74\\k74_02\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k74_02_cifar10_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=3000, batch_size=128, validation_split=0.2, callbacks=[es, mcp])
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




#### 실습 #####
# 비교할 것
#1. 이전에 본인이 한 최상의 결과와
#2. 가중치를 동결하지 않고 훈련시켰을때, trainable = True (디폴트)
#3. 가중치 동결하고 훈련시켰을 때, trainable=False
#### 위에 2,3 번 할 때는 time 체크 할 것



#1. 
#padding='same'
#loss :  0.5533019304275513
# ACC :  0.812

#2. 동결
#loss :  1.1994893550872803
#ACC :  0.585
#걸린 시간 :  96.17 초

#3. 동결x
# loss :  0.6866620182991028
# ACC :  0.795
# 걸린 시간 :  151.32 초
