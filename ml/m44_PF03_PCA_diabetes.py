from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time
import numpy as np
#1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442, )


#[실습]
#R2 0.62 이상
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=StandardScaler()
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


for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  
    #2. modeling
    from keras.layers import Dropout
    model=Sequential()
    model.add(Dense(251, input_dim=n[i]))
    model.add(Dropout(0.3))
    model.add(Dense(141))
    model.add(Dropout(0.3))
    model.add(Dense(171))
    model.add(Dropout(0.3))
    model.add(Dense(14))
    model.add(Dropout(0.3))
    model.add(Dense(5))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    # #3. compile
    model.compile(loss='mse', optimizer='adam', metrics = ['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    print(date) #2024-07-26 16:49:57.565880
    print(type(date)) #<class 'datetime.datetime'>
    date = date.strftime("%m%d_%H%M")
    print(date) #0726_1654
    print(type(date)) #<class 'str'>


    path = 'C:\\ai5\\_save\\m05\\m05_03\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path, 'm05_03', date, '_' , filename])


    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 1,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=1000, batch_size=32, validation_split=0.3, callbacks=[es, mcp])
    end_time=time.time()

    #4. predict
    loss=model.evaluate(x_test1, y_test)
    y_predict=model.predict(x_test1)

    from sklearn.metrics import r2_score
    r2=r2_score(y_test, y_predict)
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")
    print("R2의 점수 : ", r2)

#loss :  2887.905517578125
# R2의 점수 :  0.6110683426680202

# loss :  3161.851318359375
# R2의 점수 :  0.5741744184561032



#걸린시간
#cpu :  1.38초
#gpu :  4.78초

##########################################
# 결과.PCA : 7
# loss :  3506.47021484375
# 걸린 시간 :  1.97 초
# R2의 점수 :  0.5277625256638273
# 2024-08-22 13:03:07.778915

# 결과.PCA : 8
# loss :  3494.685302734375
# 걸린 시간 :  1.02 초
# R2의 점수 :  0.5293496486638898

# 결과.PCA : 9
# loss :  3519.5947265625
# 걸린 시간 :  1.0 초
# R2의 점수 :  0.5259949695316681

# 결과.PCA : 10
# loss :  3215.31640625
# 걸린 시간 :  2.32 초
# R2의 점수 :  0.5669740167899426