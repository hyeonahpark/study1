from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time

x,y = load_digits(return_X_y=True) #return_X_y= True로 작성 가능
y=pd.get_dummies(y)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 64)
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
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_dim=64))
    model.add(Dropout(0.3))
    model.add(Dense(200,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(600,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200,  activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    #3. compile

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")


    start_time=time.time()
    hist=model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=[es])
    end_time=time.time()


    #4.predict
    loss=model.evaluate(x_test, y_test, verbose=0)
    y_predict = model.predict(x_test)
    y_predict = np.round(y_predict)

    from sklearn.metrics import r2_score, accuracy_score
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")

# loss :  0.16577006876468658
# ACC :  0.956

#dropout
# loss :  0.16080109775066376
# ACC :  0.972

# loss :  0.13260766863822937
# ACC :  0.961

#걸린시간
#cpu : 1.31초
#gpu : 2.79초

###############################################################
# 결과.PCA : 30
# loss :  0.02842722274363041
# ACC :  0.994444
# 걸린 시간 :  6.58 초

# 결과.PCA : 44
# loss :  0.07948794960975647
# ACC :  0.977778
# 걸린 시간 :  7.4 초

# 결과.PCA : 55
# loss :  0.07081346958875656
# ACC :  0.983333
# 걸린 시간 :  4.48 초

# 결과.PCA : 61
# loss :  0.06829365342855453
# ACC :  0.977778
# 걸린 시간 :  3.21 초

############################### PF #######################
#결과.PCA : 1
#loss :  0.05119313672184944
#ACC :  0.983333
#걸린 시간 :  3.1 초

#결과.PCA : 1
#loss :  0.08189815282821655
#ACC :  0.972222
#걸린 시간 :  2.78 초

#결과.PCA : 1
#loss :  0.07331408560276031
#ACC :  0.972222
#걸린 시간 :  3.57 초

#결과.PCA : 64
#loss :  0.10948599874973297
#ACC :  0.983333
#걸린 시간 :  4.58 초