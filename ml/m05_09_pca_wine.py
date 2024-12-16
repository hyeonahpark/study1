from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

# print(y)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y)) 
#1    71
#0    59
#2    48

y=pd.get_dummies(y)
print(y.shape) #(178, 3)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, shuffle=True, random_state=1186, stratify=y)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
# scaler=MinMaxScaler()
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

#2. modeling
from keras.layers import Dropout
for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)  

    from keras.layers import Dropout
    model=Sequential()
    model.add(Dense(13, activation='relu', input_dim=n[i]))
    model.add(Dropout(0.3))
    # model.add(Dense(26, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(39, activation='relu'))
    # model.add(Dense(52, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(65, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(65, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(39, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(39, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(13, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))


    #3. compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")


    path_save = 'C:\\ai5\\_save\\m05\\m05_09\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_09_', str(i+1), '_', date, '_' , filename])

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)


    start_time=time.time()
    hist=model.fit(x_train1, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()


    #4. predict
    loss=model.evaluate(x_test1, y_test, verbose = 0)
    y_pred = model.predict(x_test1) 
    y_pred = np.round(y_pred)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초")


#loss :  0.16457915306091309
# ACC :  0.944

# loss :  0.36784160137176514
# ACC :  0.944

#걸린시간
#cpu : 1.94초
#gpu : 3.46초

########################################################
# 결과.PCA : 10
# loss :  0.6200529932975769
# ACC :  0.666667
# 걸린 시간 :  3.05 초

# 결과.PCA : 12
# loss :  0.4811011850833893
# ACC :  0.833333
# 걸린 시간 :  2.4 초

# 결과.PCA : 13
# loss :  0.3149016201496124
# ACC :  1.0
# 걸린 시간 :  3.07 초

# 결과.PCA : 13
# loss :  0.5533566474914551
# ACC :  0.722222
# 걸린 시간 :  3.38 초
##########################################################