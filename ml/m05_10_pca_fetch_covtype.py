from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time

#1. data

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

y=pd.get_dummies(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y) #stratify : 정확하게 train_size 비율대로 잘라줌

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 54)
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

    # 2. modeling
    from keras.layers import Dropout
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_dim=n[i]))
    model.add(Dropout(0.3))
    model.add(Dense(200,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(400,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))


    #3. compile

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path_save = 'C:\\ai5\\_save\\m05\\m05_10\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_10_', str(i+1), '_', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)

    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=50, batch_size=1024, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()


    #4.predict
    loss=model.evaluate(x_test1, y_test, verbose=0)
    y_predict = model.predict(x_test1)
    y_predict = np.round(y_predict)

    from sklearn.metrics import r2_score, accuracy_score
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")


# loss :  0.18621324002742767
# ACC :  0.945

#dropout
#loss :  0.42781001329421997
# ACC :  0.826

#걸린시간
#cpu : 119.31초
#gpu : 53.51초

##########################################################
# 결과.PCA : 25
# loss :  0.3469763398170471
# ACC :  0.860263

# 걸린 시간 :  103.05 초
# 결과.PCA : 37
# loss :  0.3107425570487976
# ACC :  0.870796
# 걸린 시간 :  102.48 초

# 결과.PCA : 46
# loss :  0.31127336621284485
# ACC :  0.873636
# 걸린 시간 :  117.11 초

# 결과.PCA : 52
# loss :  0.2977895140647888
# ACC :  0.879901
# 걸린 시간 :  116.42 초
############################################################ 