#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

#y는 T (degC), 자르는건 맘대로  / 2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00에 대한 데이터를 예측하기 (Y=144) train y.shape=(N,144) -> predict (1,144)
#일요일 23:59:59초까지, 평가지표 rmse, csv파일 제출

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Flatten, GRU
import time

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
path_data = "C:\\ai5\\_data\\kaggle\\jena\\"

dataset=pd.read_csv(path_data + "jena_climate_2009_2016.csv", index_col=0)
a = dataset.head(420407) # 예측해야하는 144개를 뺀 나머지 훈련시킬 데이터
a2 = a.tail(144) # 예측해야하는 144개의 데이터 (x_pred)
b = dataset.tail(144) #원래 정답

x_pred = a2.drop(['T (degC)'], axis=1)
# print(x_pred.shape) #(144, 13)

x=a.drop(['T (degC)'], axis=1)
y=a['T (degC)']

size = 144
def split_x(dataset, size) :
    aaa=[]
    for i in range(len(dataset)-size+1):
        subset = dataset[i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x, size)

size2 = 144
def split_y(dataset, size2) :
    aaa=[]
    for i in range(len(dataset)-size2+1):
        subset = dataset[i: (i+size2)]
        aaa.append(subset)
    return np.array(aaa)

y = split_y(y, size2)


x= np.delete(x, -1 , axis = 0)
y= np.delete(y, 0 , axis = 0)

# print(x.shape) #(420263, 144, 13)

x= x.reshape(420263, 144*13)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
# print(n) #[141, 225, 297, 1872][141, 225, 297, 1872]

for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  
    # 2. 모델 구성
    model = Sequential()
    model.add(Dense(128, input_shape=(n[i], ), activation='relu')) # timesteps , features
    model.add(Dense(256, activation='relu')) # timesteps , features
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(144))


    # #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
    start_time = time.time()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path_save = 'C:\\ai5\\_save\\m05\\m05_21\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_21_', str(i+1), '_', date, '_' , filename])
    #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
    ################## mcp 세이브 파일명 만들기 끝 ###################


    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    model.fit(x_train1, y_train, epochs=1000, batch_size=512, verbose =0, validation_split=0.2, callbacks=[es, mcp])
    end_time = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0, batch_size=512)
    x_pred=np.array([x_pred]).reshape(n[i]*13, 1)
    y_pred = model.predict(x_pred, batch_size=512)
    y_pred = np.array([y_pred]).reshape(n[i],)

    from sklearn.metrics import r2_score, mean_squared_error
    y_test = b['T (degC)']

    def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))

    rmse = RMSE(y_test, y_pred)
    
    print("##############################################")
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수
    print("RMSE :", rmse)

# ###csv 파일 만들기 ###
# submit = pd.read_csv(path_data + "jena_climate_2009_2016.csv")

# submit = submit[['Date Time','T (degC)']]
# submit = submit.tail(144)
# # print(submit)

# # y_submit = pd.DataFrame(y_predict)
# # print(y_submit)

# submit['T (degC)'] = y_pred
# # print(submit)                  # [6493 rows x 1 columns]
# # print(submit.shape)            # (6493, 1)

# submit.to_csv(path + "jena_박현아.csv", index = False)



#RMSE : 1.1558566105864292, 0.23490005731582642, batch 512, unit 128 시작
