#https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv

#scaling 작업 -> 몰려있는 값 표준편차로 구하기

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd

#1. data

path = 'C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)  # (165034, 13)
print(test_csv.shape)  # (110023,12)
print(sample_submission.shape) #(110023,1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

x=train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
test_csv=test_csv.drop(['CustomerId', 'Surname'], axis=1)

y=train_csv['Exited']


from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])
test_csv[:] = scalar.fit_transform(test_csv[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)
# print(x_train.shape, y_train.shape) #(148530, 10) (148530,)

# gbrt = GradientBoostingClassifier(random_state=0)
# gbrt.fit(x_train, y_train)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
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
    model.add(Dense(32, input_dim=n[i], activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))


    #3. compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path_save = 'C:\\ai5\\_save\\m05\\m05_08\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_08_', str(i+1), '_', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=50, batch_size=1000, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()

    #4. predict
    loss=model.evaluate(x_test1, y_test, verbose = 0)
    y_predict = model.predict(x_test1)
    y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.

    from sklearn.metrics import r2_score, accuracy_score
    r2 = r2_score(y_test, y_predict)
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")

#loss :  0.3242044150829315
# ACC :  0.862

#dropout
# loss :  0.38874727487564087
# ACC :  0.859


#걸린시간
#cpu : 28.32초 
#gpu : 153.74초

##########################################
# 결과.PCA : 8
# loss :  0.45679667592048645
# ACC :  0.79338
# 걸린 시간 :  9.77 초

# 결과.PCA : 10
# loss :  0.4298621714115143
# ACC :  0.79338
# 걸린 시간 :  8.1 초

# 결과.PCA : 10
# loss :  0.44638967514038086
# ACC :  0.79338
# 걸린 시간 :  11.31 초


# 결과.PCA : 10
# loss :  0.4254995286464691
# ACC :  0.79338
# 걸린 시간 :  9.68 초
