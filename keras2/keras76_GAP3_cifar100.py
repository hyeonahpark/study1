#keras 76_GAP3_cifar100
#keras 76_GAP4_horse
#keras 76_GAP5_rps
#keras 76_GAP6_kaggle_cat_dog
#keras 76_GAP7_men_women

#GAP과 Flatten 비교
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
from tensorflow.keras.layers import Flatten, Dropout, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
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
#vgg16.trainable = True # false : 가중치 동결


#2. modeling
model=Sequential()
model.add(vgg16)
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))
model.summary()

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
#print(y_predict)


from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("ACC score :", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# flatten
# 동결x
# loss :  2.6391401290893555
# ACC :  0.342
# 걸린 시간 :  140.47 초

# GAP
# loss :  2.303164005279541
# ACC :  0.1
# ACC score : 1.0