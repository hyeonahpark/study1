#m04_1에서 뽑은 4가지 결과로 4가지 모델을 맹그러
# input_shape=()
# 1. 70000,154
# 2. 70000,331
# 3. 70000,486
# 4. 70000,713
# 5. 70000,784

#시간과 성능 체크

#결과 예시
#결과 1. pca =154
#걸린시간 000초
# acc=000

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1.data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x_train=x_train/255.
x_test=x_test/255.

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])


pca = PCA(n_components = 28*28)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
# print(sum(evr))
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.fit_transform(y_test.reshape(-1,1))

# print("0.95이상 : ",np.argmax(evr_cumsum>=0.95)+1) #154
# print("0.99이상 : ",np.argmax(evr_cumsum>=0.99)+1) #331
# print("0.999이상 : ",np.argmax(evr_cumsum>=0.999)+1) #486
# print("1.0 : ", np.argmax(evr_cumsum>=1.0)+1) #713

for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  
    #2. modeling
    model= Sequential()
    model.add(Dense(128, input_shape=(n[i], )))
    model.add(Dense(256,  activation = 'relu'))
    model.add(Dense(512,  activation = 'relu'))
    model.add(Dense(1024,  activation = 'relu'))
    model.add(Dense(512,  activation = 'relu'))
    model.add(Dense(256,  activation = 'relu'))
    model.add(Dense(128,  activation = 'relu'))
    model.add(Dense(10, activation='softmax'))


    #3. compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    print(date) #2024-07-26 16:49:57.565880
    print(type(date)) #<class 'datetime.datetime'>
    date = date.strftime("%m%d_%H%M")
    print(date) #0726_1654
    print(type(date)) #<class 'str'>


    path = 'C:\\ai5\\_save\\m04\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path, 'm04_', date, '_' , filename])
    #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
    ################## mcp 세이브 파일명 만들기 끝 ###################

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 1,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()

    # model.save('./_save/keras35/keras35_04_mcp.hdf5')

    #4. predict

    loss=model.evaluate(x_test1, y_test)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)
    y_predict = model.predict(x_test1)
    y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
    # print(y_predict)

    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 3))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

############################################################
# batch 128
# 결과.PCA : 154
# loss :  0.0862678736448288
# ACC :  0.974
# 걸린 시간 :  10.45 초

# 결과.PCA : 331
# loss :  0.09376537054777145
# ACC :  0.971
# 걸린 시간 :  10.65 초

# 결과.PCA : 486
# loss :  0.10224473476409912
# ACC :  0.97
# 걸린 시간 :  11.68 초

# 결과.PCA : 713
# loss :  0.11188653856515884
# ACC :  0.967
# 걸린 시간 :  11.43 초
##########################################################
# batch 32
#결과.PCA : 154
# loss :  0.09110896289348602
# ACC :  0.977
# 걸린 시간 :  58.9 초

# 결과.PCA : 331
# loss :  0.10910484939813614
# ACC :  0.976
# 걸린 시간 :  86.2 초

# 결과.PCA : 486
# loss :  0.12022645026445389
# ACC :  0.972
# 걸린 시간 :  64.12 초

# 결과.PCA : 713
# loss :  0.12538208067417145
# ACC :  0.971
# 걸린 시간 :  69.49 초


