#cpu일 때와 cpu일 때의 시간 비교
import time



import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model #load_model : model 을 불러옴
from tensorflow.keras.layers import Dense, Input
import sklearn as sk
print(sk.__version__) #0.24.2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

#1.data
dataset=load_boston()

x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, random_state=6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 13)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
evr = pca.explained_variance_ratio_ # 설명가능한 변화율
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  
    # #2. modeling
    model=Sequential()
    model.add(Dense(64, input_shape=(n[i], ))) # 특성은 항상 많으면 좋음! 데이터가 많으면 좋으니까
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(Dropout(0.3))
    model.add(Dense(16))
    model.add(Dropout(0.3))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    #3. compile
    model.compile(loss='mse', optimizer='adam', metrics = ['acc'])

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


    path = 'C:\\ai5\\_save\\m05\\m05_01\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path, 'm05_01', date, '_' , filename])
    #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
    ################## mcp 세이브 파일명 만들기 끝 ###################

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 1,
        save_best_only=True,
        filepath=filepath)

    start_time=time.time() #time.time() 현재 시간 반환
    hist = model.fit(x_train1, y_train, epochs=1000, batch_size=32, verbose = 1, validation_split=0.2, callbacks=[es, mcp]) #hist는 history의 약자,
    end_time=time.time() #끝나는 시간 반환

    #4. predict

    loss=model.evaluate(x_test1, y_test)
    y_predict=model.predict(x_test1)
    from sklearn.metrics import r2_score
    r2=r2_score(y_test, y_predict)
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 3))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")
    print("R2의 점수 : ", r2)


# #loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718
# 걸린 시간 :  1.97 초

# # ===================1. save.model 출력 =========================
# loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718
# ===================2. mcp 출력 =========================
# loss :  28.253427505493164
# R2의 점수 :  0.7397246085125718


#dropout============================================
# loss :  26.18733024597168
# R2의 점수 :  0.7587578781411475


#걸린시간
#cpu : 12.79초
#gpu : 59.15초


#####################################################
# 결과.PCA : 8
# loss :  29.251937866210938
# 걸린 시간 :  1.71 초
# R2의 점수 :  0.7648571805548144

# 결과.PCA : 12
# loss :  24.875675201416016
# 걸린 시간 :  1.09 초
# R2의 점수 :  0.8000359276360193

# 결과.PCA : 13
# loss :  31.00921058654785
# 걸린 시간 :  1.11 초
# R2의 점수 :  0.7507312779949286

#결과.PCA : 13
# loss :  25.493915557861328
# 걸린 시간 :  1.09 초
# R2의 점수 :  0.7950661772803