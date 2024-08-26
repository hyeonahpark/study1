# 배치를 160으로 잡고
# x,y를 추출해서 모델을 맹그러라
# acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

#1. data (시간체크)
np_path = 'c:/ai5/_data/_save_npy/rps/'
x_train=np.load(np_path + 'keras44_03_x_train.npy')
y_train=np.load(np_path + 'keras44_03_y_train.npy')

# print(x_train.shape) #(1100, 300, 300, 3)

x = x_train.reshape(1100,300*300*3)
y = y_train

# print(x.shape) #(1100, 90000, 3)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=5656)


from sklearn.decomposition import PCA
pca = PCA(n_components = None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.transform(x_test)  



    model = Sequential()
    model.add(Dense(64, input_shape=(n[i], ), activation='relu')) 
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(3, activation='softmax')) 
                                            
                            
    #3. compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
    start_time=time.time()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")


    path_save = 'C:\\ai5\\_save\\m05\\m05_20\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_20_', str(i+1), '_', date, '_' , filename])
    #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
    ################## mcp 세이브 파일명 만들기 끝 ###################

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)

    hist=model.fit(x_train2, y_train, epochs=1000, batch_size=10, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()


    #4. predict
    from sklearn.metrics import r2_score, accuracy_score
    loss = model.evaluate(x_test2, y_test, verbose=1)
    y_pre = np.round(model.predict(x_test2))
    print("##############################################")
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

#loss : 0.04563869908452034
# acc : 0.98182

# loss : 0.00012304952542763203
# acc : 1.0

#conv1d
# loss : 0.21916770935058594
# acc : 0.95455
# 걸린 시간 : 1162.15 초

##############################################
# 결과.PCA : 133
# loss :  0.030501997098326683
# ACC :  0.981818
# 걸린 시간 :  8.03 초
# 4/4 [==============================] - 0s 2ms/step - loss: 0.0049 - accuracy: 1.0000 - acc: 1.0000 - mse: 1.0254e-04
# ##############################################
# 결과.PCA : 366
# loss :  0.004863070789724588
# ACC :  1.0
# 걸린 시간 :  9.16 초
# 4/4 [==============================] - 0s 2ms/step - loss: 0.0156 - accuracy: 0.9909 - acc: 0.9909 - mse: 0.0034
# ##############################################
# 결과.PCA : 690
# loss :  0.015555704943835735
# ACC :  0.990909
# 걸린 시간 :  9.46 초
# 4/4 [==============================] - 0s 2ms/step - loss: 0.0336 - accuracy: 0.9909 - acc: 0.9909 - mse: 0.0057
# ##############################################
# 결과.PCA : 988
# loss :  0.033557284623384476
# ACC :  0.990909
# 걸린 시간 :  9.42 초