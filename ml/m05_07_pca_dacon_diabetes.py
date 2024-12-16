import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping


#1. data
datasets = load_breast_cancer()
# # print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data #(569, 30)
y = datasets.target #(569, )
# print(x.shape, y.shape) 
# print(type(x)) # <class 'numpy.ndarray'> 

# numpy, pandas에서 y의 라벨 종류를 찾아낼 수 있음
# numpy로 y의 종류와 개수 파악  numpy.unique / pandas.valueCount
# 0과 1의 갯수가 몇개인지 찾기
unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

# print("고유한 요소:", unique) #고유한 요소: [0 1]
# print("각 요소의 개수:", counts) #각 요소의 개수: [212 357]

# print(pd.DataFrame(y).value_counts())
# # 1    357
# # 0    212
# print(pd.Series(y).value_counts)
# print(pd.value_counts(y)) #1    357 / 0    212


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=6666)

# print(x_train.shape, y_train.shape) # (398, 30) (398, )
# print(x_test.shape, y_test.shape) # (171, 30) (171, )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 30)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
evr = pca.explained_variance_ratio_ # 설명가능한 변화율
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

#2. modeling
from keras.layers import Dropout
for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  
    from keras.layers import Dropout

    model=Sequential()
    model.add(Dense(32, activation='relu', input_dim=n[i]))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid')) #최종 아웃풋 노드는 0과 1이 나와야 함. activation(한정함수, 활성화함수)를 사용하여 값을 0~1사이로 한정시킴 


    #3. compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    # print(date) #2024-07-26 16:49:57.565880
    # print(type(date)) #<class 'datetime.datetime'>
    date = date.strftime("%m%d_%H%M")
    # print(date) #0726_1654
    # print(type(date)) #<class 'str'>


    path_save = 'C:\\ai5\\_save\\m05\\m05_07\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_07_', str(i+1), '_', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=1000, batch_size=32, validation_split=0.3, verbose = 0, callbacks=[es, mcp])
    end_time=time.time()

    #4. predict

    loss=model.evaluate(x_test1, y_test, verbose = 0)
    y_pred = model.predict(x_test1) 
    y_pred = np.round(y_pred)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
    # print(y_pred)

    from sklearn.metrics import r2_score
    r2=r2_score(y_test, y_pred)
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 5))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")

#acc_score :  0.9824561403508771
# 걸린 시간 :  1.25 초

#dropout
#acc_score :  0.9883040935672515
# 걸린 시간 :  1.8 초


#걸린시간
#cpu : 1.47초
#gpu : 5.02초

###################################################
# 결과.PCA : 10
# loss :  0.08417262881994247
# ACC :  0.971
# 걸린 시간 :  3.15 초

# 결과.PCA : 17
# loss :  0.07981853932142258
# ACC :  0.977
# 걸린 시간 :  1.6 초

# 결과.PCA : 25
# loss :  0.07736258208751678
# ACC :  0.982
# 걸린 시간 :  2.32 초

# 결과.PCA : 30
# loss :  0.07161126285791397
# ACC :  0.988
# 걸린 시간 :  2.08 초
###################################################