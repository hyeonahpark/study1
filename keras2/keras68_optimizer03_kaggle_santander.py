#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)


x=train_csv.drop(['target'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
y=train_csv['target']


unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True)) #이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1], dtype=int64), array([179902,  20098], dtype=int64))


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1186)

# print(x_train.shape, y_train.shape) # (180000, 200) (180000,)
# print(x_test.shape, y_test.shape) # (20000, 200) (20000,)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. modeling
from keras.layers import Dropout
for i in range(0, len(lr), 1) :
    from keras.layers import Dropout
    model=Sequential()
    model.add(Dense(400, activation='relu', input_dim=200))
    model.add(Dropout(0.3))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid')) #최종 아웃풋 노드는 0과 1이 나와야 함. activation(한정함수, 활성화함수)를 사용하여 값을 0~1사이로 한정시킴 


    #3. compile
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['accuracy', 'acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")



    path_save = 'C:\\ai5\\_save\\keras68\\k68_03\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'k68_03_', str(i+1), '_', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train, y_train, epochs=50, batch_size=2000, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()

    #4. predict

    loss=model.evaluate(x_test, y_test, verbose = 0)

    # result = model.predict(test_csv)
    # result = np.round(result)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.

    from sklearn.metrics import r2_score, accuracy_score
    print("######################################")
    print("결과.lr :", lr[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")


# loss :  0.23717275261878967
# ACC :  0.912

#dropout
# loss :  0.24040205776691437
# ACC :  0.911

#걸린시간
#cpu : 28.17초
#gpu : 4.32초

# ######################################
# 결과.lr : 0.1
# loss :  0.32841357588768005
# ACC :  0.8985
# 걸린 시간 :  5.58 초
# ######################################
# 결과.lr : 0.01
# loss :  0.3284611403942108
# ACC :  0.8985
# 걸린 시간 :  6.77 초
# ######################################
# 결과.lr : 0.005
# loss :  0.24184951186180115
# ACC :  0.9104
# 걸린 시간 :  6.23 초
# ######################################
# 결과.lr : 0.001
# loss :  0.23945876955986023
# ACC :  0.9109
# 걸린 시간 :  10.3 초
# ######################################
# 결과.lr : 0.0005
# loss :  0.23719589412212372
# ACC :  0.91115
# 걸린 시간 :  18.69 초
# ######################################
# 결과.lr : 0.0001
# loss :  0.23947717249393463
# ACC :  0.910275
# 걸린 시간 :  17.98 초
