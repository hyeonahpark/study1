import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping


#1. data
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data #(569, 30)
y = datasets.target #(569, )
print(x.shape, y.shape) 
print(type(x)) # <class 'numpy.ndarray'> 

# numpy, pandas에서 y의 라벨 종류를 찾아낼 수 있음
# numpy로 y의 종류와 개수 파악  numpy.unique / pandas.valueCount
# 0과 1의 갯수가 몇개인지 찾기
unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

print("고유한 요소:", unique) #고유한 요소: [0 1]
print("각 요소의 개수:", counts) #각 요소의 개수: [212 357]

print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts)
print(pd.value_counts(y)) #1    357 / 0    212


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=6666)

print(x_train.shape, y_train.shape) # (398, 30) (398, )
print(x_test.shape, y_test.shape) # (171, 30) (171, )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. modeling
model=Sequential()
model.add(Dense(32, activation='relu', input_dim=30))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #최종 아웃풋 노드는 0과 1이 나와야 함. activation(한정함수, 활성화함수)를 사용하여 값을 0~1사이로 한정시킴 



#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time() #time.time() 현재 시간 반환

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=30,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, verbose = 1, validation_split=0.3,  callbacks=[es]) #hist는 history의 약자,
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
print("acc_score : ", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# mse 사용
#acc_score :  0.6140350877192983, 걸린 시간 :  1.34 초
#acc_score :  0.935672514619883 , 걸린 시간 :  5.68 초

# binary_cross entropy 사용
#acc_score :  0.935672514619883 , 걸린 시간 :  4.9 초
#acc_score :  0.9415204678362573, 걸린 시간 :  4.69 초
#acc_score :  0.9473684210526315, 걸린 시간 :  5.35 초


#minmaxscaler
#acc_score :  0.9766081871345029, 걸린 시간 :  2.69 초

#standardscaler
#acc_score :  0.9883040935672515, 걸린 시간 :  1.59 초

#maxabs
#acc_score :  0.9766081871345029

#robust
#acc_score :  0.9824561403508771