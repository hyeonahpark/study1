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

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

#### 스케일링 1-1 ######
x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.fit_transform(y_test.reshape(-1,1))

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  
    #2. modeling
    model=Sequential()
    model.add(Dense(100, input_shape=(n[i], )))
    model.add(Dense(200,  activation = 'relu'))
    model.add(Dense(300,  activation = 'relu'))
    model.add(Dense(600,  activation = 'relu'))
    model.add(Dense(300,  activation = 'relu'))
    model.add(Dense(200,  activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(10, activation='softmax'))


    #3. compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")


    path_save = 'C:\\ai5\\_save\\m05\\m05_16\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_16_', str(i+1), '_', date, '_' , filename])


    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=1000, batch_size=2000, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()


    #4. predict
    loss=model.evaluate(x_test1, y_test, verbose=0)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)
    y_predict = model.predict(x_test1)
    y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)

    print("##############################################")
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  1.4850677251815796
# ACC :  0.479
# 걸린 시간 :  148.31 초


##############################################
# 결과.PCA : 217
# loss :  1.41213059425354
# ACC :  0.5072
# 걸린 시간 :  2.11 초
# ##############################################
# 결과.PCA : 658
# loss :  1.4338703155517578
# ACC :  0.5049
# 걸린 시간 :  2.01 초
# ##############################################
# 결과.PCA : 1430
# loss :  1.456485390663147
# ACC :  0.4986
# 걸린 시간 :  2.85 초
# ##############################################
# 결과.PCA : 3072
# loss :  1.4850273132324219
# ACC :  0.4808
# 걸린 시간 :  4.11 초