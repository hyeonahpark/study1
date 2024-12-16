import numpy as np
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time

#1. data
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(20640, 8) (20640, )

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=3, )


from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
# y_train= ohe.fit_transform(y_train.reshape(-1,1))
# y_test= ohe.transform(y_test.reshape(-1,1))

for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  


    #2. modeling
    from keras.layers import Dropout
    model=Sequential()
    model.add(Dense(10, input_dim=n[i]))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    model.add(Dense(1))


    #3. compile
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


    path = 'C:\\ai5\\_save\\m05\\m05_02\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path, 'm05_02', date, '_' , filename])

    mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=path + 'best_model',
    save_format='tf'  # TensorFlow SavedModel 포맷 사용
)


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

##기존#############################################
# 결과.PCA : 3
# loss :  0.6104937195777893
# 걸린 시간 :  52.83 초
# R2의 점수 :  0.5469272645792587

# 결과.PCA : 4
# loss :  0.5312401652336121
# 걸린 시간 :  32.39 초
# R2의 점수 :  0.6057446253346895

# 결과.PCA : 6
# loss :  0.5237132906913757
# 걸린 시간 :  51.04 초
# R2의 점수 :  0.6113306986545679

# 결과.PCA : 8
# loss :  0.5072473287582397
# 걸린 시간 :  103.5 초
# R2의 점수 :  0.6235507003006264
###############################################

################# PF ##########################
#결과.PCA : 7
#loss :  0.5232160687446594
#걸린 시간 :  56.57 초
#R2의 점수 :  0.6116996782673441

#결과.PCA : 8
#loss :  0.5221644043922424
#걸린 시간 :  46.88 초
#R2의 점수 :  0.6124802569724537