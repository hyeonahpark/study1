"""
keras49_augument2_mnist.py
keras49_augument3_cifar10.py
keras49_augument4_cifar100.py
keras49_augument5_cat_dog.py
keras49_augument6_men_women.py
keras49_augument7_horse.py
keras49_augument8_rps.py
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import time
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

train_datagen =  ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    # horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=0.2,              # 축소 또는 확대
    shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

augment_size = 40000 
 
print(x_train.shape[0])  #60000
randidx=np.random.randint(x_train.shape[0], size = augment_size) # 이미지 1장 # 60000, size = 40000
print(randidx) # [31998 12497 32753 ... 32228 21276 28073]
print(np.min(randidx), np.max(randidx)) # randidx의 최솟값과 최댓값 출력 1, 59991 

print(x_train[0].shape) # (28, 28)

x_augmented = x_train[randidx].copy() #.copy하면 메모리를 따로 할당하므로 원래 있던 x_train 값에 영향을 미치지 않음.
y_augmented = y_train[randidx].copy() 
print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)
 
x_augmented = x_augmented.reshape(
    x_augmented.shape[0], #40000
    x_augmented.shape[1], #28
    x_augmented.shape[2], 1 #28
)

print(x_augmented.shape) #(40000, 28, 28, 1)

 
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/ai5/_data/_save_img/02_mnist/'
).next()[0]


# print(x_augmented.shape) # ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape', (40000, 28, 28))
# print(x_augmented.shape) # (40000, 28, 28, 1)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)
 
x_train = np.concatenate((x_train, x_augmented))
print(x_train.shape) #(100000, 28, 28, 1))

y_train = np.concatenate((y_train, y_augmented))
print(y_train.shape) #(100000,)

# y_train=pd.get_dummies(y_train)
# y_test=pd.get_dummies(y_test)
# # ##oneHot 1-3
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.fit_transform(y_test.reshape(-1,1))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000, 10)
print(x_test.shape, y_test.shape) # (10000, 28, 28, 1) (10000, 10)


#2. modeling
model=Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time()
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


path = 'C:\\ai5\\_save\\keras49\\k49_02_mnist\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k49_02_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)



hist=model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.3, callbacks=[es, mcp])
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


#지난번
#loss :  0.05474101006984711
#ACC :  0.985

#augment
# loss :  0.029937705025076866
# ACC :  0.994
