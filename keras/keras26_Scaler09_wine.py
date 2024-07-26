from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, shuffle=True, random_state=6666, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# #2. modeling
model=Sequential()
model.add(Dense(13, activation='relu', input_dim=13))
# model.add(Dense(26, activation='relu'))
model.add(Dense(39, activation='relu'))
# model.add(Dense(52, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time() #time.time() 현재 시간 반환

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=50,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose = 1, validation_split=0.1,  callbacks=[es]) #hist는 history의 약자,
end_time=time.time() #끝나는 시간 반환

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 1)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))

y_pred = model.predict(x_test) 
y_pred = np.round(y_pred)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
print(y_pred)

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  0.22040227055549622, ACC :  0.944, 걸린 시간 :  18.28 초
#loss :  0.2100989669561386, ACC :  0.944, 걸린 시간 :  14.74 초
#loss :  0.17500953376293182, ACC :  0.944, 걸린 시간 :  9.12 초
#loss :  0.22056569159030914. ACC :  0.944, 걸린 시간 :  15.6 초

#==============================================minmax
#loss :  0.0 , ACC :  1.0

#==============================================standard
#loss :  0.008300263434648514 ACC :  1.0

#maxabs
#loss :  0.14651073515415192 ACC :  0.944

#robust
#loss :  0.03531860560178757, ACC :  1.0