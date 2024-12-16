#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time

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
    model=Sequential()
    model.add(Dense(400, activation='relu', input_dim=n[i]))
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")



    path_save = 'C:\\ai5\\_save\\m05\\m05_12\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_12_', str(i+1), '_', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=50, batch_size=2000, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()

    #4. predict

    loss=model.evaluate(x_test1, y_test, verbose = 0)

    # result = model.predict(test_csv)
    # result = np.round(result)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.

    from sklearn.metrics import r2_score, accuracy_score
    print("######################################")
    print("결과.PCA :", n[i])
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
# 결과.PCA : 1
# loss :  0.3284367322921753
# ACC :  0.8985
# 걸린 시간 :  5.5 초
# ######################################
# 결과.PCA : 1
# loss :  0.32808107137680054
# ACC :  0.8985
# 걸린 시간 :  5.6 초
# ######################################
# 결과.PCA : 1
# loss :  0.32841387391090393
# ACC :  0.8985
# 걸린 시간 :  4.54 초
# ######################################
# 결과.PCA : 64
# loss :  0.26914969086647034
# ACC :  0.902675
# 걸린 시간 :  11.85 초