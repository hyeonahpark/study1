# 배치를 160으로 잡고
# x,y를 추출해서 모델을 맹그러라
# acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

#1. data (시간체크)
path = './_data/image/horse_human/'
# path_test = './_data/image/cat_and_dog/Test/'

np_path = 'c:/ai5/_data/_save_npy/horse/'
x=np.load(np_path + 'keras44_02_x_train.npy')
y=np.load(np_path + 'keras44_02_y_train.npy')

# print(x.shape) #(1027, 200, 200, 3)

x = x.reshape(1027,200*200*3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=5656)

for i in range(0, len(lr), 1) :

    model = Sequential()
    model.add(Dense(64, input_shape=(200*200*3, ), activation='relu')) 
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) 
                                            
                            
    #3. compile
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['accuracy', 'acc', 'mse'])
    start_time=time.time()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")


    path_save = 'C:\\ai5\\_save\\keras68\\k68_09\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'k68_09_', str(i+1), '_', date, '_' , filename])
    #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
    ################## mcp 세이브 파일명 만들기 끝 ###################

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)

    hist=model.fit(x_train, y_train, epochs=1000, batch_size=1024, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()


    #4. predict
    from sklearn.metrics import r2_score, accuracy_score
    loss = model.evaluate(x_test, y_test, verbose=1)
    y_pre = np.round(model.predict(x_test))
    print("##############################################")
    print("결과.lr :", lr[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수
   

# loss : 0.0006491452804766595
# acc : 1.0



#cnn1d
# loss : 0.2974087595939636
# acc : 0.95146
# 걸린 시간 : 85.17 초

# ##############################################
# 결과.lr : 0.1
# loss :  6.909980297088623
# ACC :  0.475728
# 걸린 시간 :  3.48 초
# 4/4 [==============================] - 0s 4ms/step - loss: 1.8329 - accuracy: 0.4757 - acc: 0.4757 - mse: 0.4734
# ##############################################
# 결과.lr : 0.01
# loss :  1.8329176902770996
# ACC :  0.475728
# 걸린 시간 :  3.43 초
# 4/4 [==============================] - 0s 4ms/step - loss: 4.9535 - accuracy: 0.5243 - acc: 0.5243 - mse: 0.4749
# ##############################################
# 결과.lr : 0.005
# loss :  4.953493118286133
# ACC :  0.524272
# 걸린 시간 :  2.37 초
# 4/4 [==============================] - 0s 4ms/step - loss: 0.6600 - accuracy: 0.5631 - acc: 0.5631 - mse: 0.2338
# ##############################################
# 결과.lr : 0.001
# loss :  0.6600034236907959
# ACC :  0.563107
# 걸린 시간 :  2.87 초
# 4/4 [==============================] - 0s 7ms/step - loss: 0.1327 - accuracy: 0.9515 - acc: 0.9515 - mse: 0.0379
# ##############################################
# 결과.lr : 0.0005
# loss :  0.13269579410552979
# ACC :  0.951456
# 걸린 시간 :  38.61 초
# 4/4 [==============================] - 0s 5ms/step - loss: 0.1026 - accuracy: 0.9612 - acc: 0.9612 - mse: 0.0323
# ##############################################
# 결과.lr : 0.0001
# loss :  0.10256306082010269
# ACC :  0.961165
# 걸린 시간 :  62.45 초

