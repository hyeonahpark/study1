#https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

start_time=time.time()
np_path = 'c:/ai5/_data/_save_npy/biggest_gender2/'
x_train1=np.load(np_path + 'keras45_07_x_train1.npy')
y_train1=np.load(np_path + 'keras45_07_y_train1.npy')
x_train2=np.load(np_path + 'keras45_07_x_train2.npy')
y_train2=np.load(np_path + 'keras45_07_y_train2.npy')
x_test2=np.load(np_path + 'keras45_07_x_test2.npy')
y_test2=np.load(np_path + 'keras45_07_y_test2.npy')

x_train= np.concatenate((x_train1[:5000],x_train2))
y_train= np.concatenate((y_train1[:5000],y_train2))

end_time=time.time()
print("데이터 불러오는 시간 :", round(end_time-start_time,2),'초') 

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=5656)

#2. modeling
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(#weights='imagenet',
              include_top=False,
              input_shape=(100,100,3)
)
vgg16.trainable = False # 가중치 동결

#2. modeling
model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

                                          
                 
#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras74\\k74_07\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k74_07_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ################### 

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, mcp])


model=load_model('_save/keras45/k45_07/k45_07_0813_2126_0026-0.2104.hdf5')

end_time=time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

# y_pre = np.round(model.predict(x_test, batch_size=16))
print("걸린 시간 :", round(end_time-start_time,2),'초')


y_submit = model.predict(x_test2, batch_size=16)
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
#print(y_submit)


"""
1.
loss : 0.19363650679588318
acc : 0.92494
걸린 시간 : 1.8 초

2. 동결




3. 동결 x
loss : 0.19363650679588318
acc : 0.92494
걸린 시간 : 320.24 초


"""