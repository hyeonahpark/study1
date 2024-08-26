#https://dacon.io/competitions/official/236068

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd


#1. data

path = 'C:\\ai5\\_data\\dacon\\diabetes\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)  # (652,9)
# print(test_csv.shape)  # (116,8)
# print(sample_submission.shape) #(116,1)

################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

# ###########################################

# print(test_csv.info())

# print(train_csv.describe())

################# x와 y 분리 ######################
x=train_csv.drop(['Outcome'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
# print(x)
# print(x.shape) #(652, 8)
y=train_csv['Outcome']
# print(y.shape) # (652., )

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

# print("고유한 요소:", unique) #고유한 요소: [0 1]
# print("각 요소의 개수:", counts) #각 요소의 개수: [424 228]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
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

    #2. modeling
    from keras.layers import Dropout
    model=Sequential()
    model.add(Dense(16, input_dim=n[i], activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
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
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))


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


    path_save = 'C:\\ai5\\_save\\m05\\m05_06\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_06', '_', str(i+1), date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=1000, batch_size=32, verbose = 0, validation_split=0.3, callbacks=[es, mcp])
    end_time=time.time()


    #4. predict
    loss=model.evaluate(x_test1, y_test, verbose = 0)
    y_predict = model.predict(x_test1)
    y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.

    from sklearn.metrics import r2_score
    r2=r2_score(y_test, y_predict)
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 3))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")
    # print("R2의 점수 : ", r2)

#dropout
#loss :  0.5292198061943054
# ACC :  0.652

#걸린시간
#cpu : 1.03초
#gpu : 2.1초
######################################################

# 결과.PCA : 7
# loss :  0.48855867981910706
# ACC :  0.636
# 걸린 시간 :  5.41 초
# 결과.PCA : 8
# loss :  0.4593880772590637
# ACC :  0.712
# 걸린 시간 :  5.69 초
# 결과.PCA : 8
# loss :  0.46966269612312317
# ACC :  0.667
# 걸린 시간 :  4.74 초
# 결과.PCA : 8
# loss :  0.5635251998901367
# ACC :  0.636
# 걸린 시간 :  3.62 초