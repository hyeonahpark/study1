import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. data
datasets = load_iris()
# print(datasets) # column 수 : 4개
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,0)

print(y)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([50, 50, 50], dtype=int64))

print(pd.value_counts(y)) 
#0    50
#1    50
#2    50


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, shuffle=True, random_state=1186, stratify=y )
print(pd.value_counts(y_train))
#2    48
# 0    44
# 1    43



####[실습]#### pandas.get_dummies , sklearn.preprocessing.OneHotEncoder, keras  
# print(x)
#원핫 : 판다스
y=pd.get_dummies(y)


#원핫 : 사이킷런 - fit 하고 transform
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
# y= ohe.fit_transform(y.reshape(-1,1)) #(-1,1)은 전체를 뜻함 -1은 데이터의 끝을 의미하기 때문 
#reshape 조건 1. 데이터의 내용(값)이 바뀌면 안된다. 2. 데이터의 순서가 바뀌면 안된다.

#원핫 : 케라스
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)

print(y.shape) #(150, 3)
print(y)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, shuffle=True, random_state=1186, stratify=y)

# #2. modeling
model=Sequential()
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time() #time.time() 현재 시간 반환

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=100,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose = 1, validation_split=0.1,  callbacks=[es]) #hist는 history의 약자,
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




#linear 
#loss :  0.021191222593188286, ACC :  0.867. 걸린 시간 :  3.24 초
#softmax
#loss :  0.015708258375525475, ACC :  1.0, 걸린 시간 :  4.33 초
