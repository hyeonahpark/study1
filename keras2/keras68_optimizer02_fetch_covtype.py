from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)

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

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

#2. modeling
from keras.layers import Dropout
for i in range(0, len(lr), 1) :

    # 2. modeling
    from keras.layers import Dropout
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_dim=54))
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
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='categorical_crossentropy', optimizer = Adam(learning_rate=lr[i]), metrics=['accuracy','acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path_save = 'C:\\ai5\\_save\\keras68\\k68_02\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'k68_02_', str(i+1), '_', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)

    start_time=time.time()
    hist=model.fit(x_train, y_train, epochs=50, batch_size=1024, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()


    #4.predict
    loss=model.evaluate(x_test, y_test, verbose=0)
    y_predict = model.predict(x_test)
    y_predict = np.round(y_predict)

    from sklearn.metrics import r2_score, accuracy_score
    print("결과.lr :", lr[i])
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
############################################
# 결과.lr : 0.1
# loss :  1.2053909301757812
# ACC :  0.487608
# 걸린 시간 :  10.79 초

# 결과.lr : 0.01
# loss :  0.6295526027679443
# ACC :  0.735259
# 걸린 시간 :  24.41 초

# 결과.lr : 0.005
# loss :  0.48900437355041504
# ACC :  0.800041
# 걸린 시간 :  50.06 초

# 결과.lr : 0.001
# loss :  0.34738799929618835
# ACC :  0.858525
# 걸린 시간 :  93.89 초

# 결과.lr : 0.0005
# loss :  0.3399399518966675
# ACC :  0.860074
# 걸린 시간 :  96.72 초

# 결과.lr : 0.0001
# loss :  0.4398011863231659
# ACC :  0.816306
# 걸린 시간 :  88.97 초
#############################################