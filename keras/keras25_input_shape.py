#18-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
print(sk.__version__) #0.24.2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time

#1.data
dataset=load_boston()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']


# print(dataset)
x=dataset.data
y=dataset.target


# print(x)
# print(x.shape) #(506,13)
# print(y)
# print(y.shape) #(506, )

#train_size : 0.7~0.9 사이로
#R2 0.8 이상
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=6666)

#2. modeling
model=Sequential()
# model.add(Dense(10, input_dim=13)) # 특성은 항상 많으면 좋음! 데이터가 많으면 좋으니까
model.add(Dense(10, input_shape=(13, ))) # 이미지 input_shape=(8,8,1)
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(1))

# 한계 : 항상 input_dim은 행렬 형태였음. 하지만 다차원 행렬의 경우에는 input.shape를 해줘야 함.
# 이미지에서 8*8이 100개 있는 경우 100*8*8 => 100*64로 변경 가능 / 컬러인 경우는 100*8*8*3 (RGB)


#3. compile
model.compile(loss='mse', optimizer='adam')
start_time=time.time() #time.time() 현재 시간 반환
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose = 1, validation_split=0.2) #hist는 history의 약자,
end_time=time.time() #끝나는 시간 반환

#4. predict

loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)
print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

print("=================================hist======================================")
print(hist) 
print("==============================hist.history======================================")
print(hist.history) #loss와 val_loss가 epochs 수 만큼 출력됨
print("==============================loss======================================")
print(hist.history['loss']) #history에서 loss 값만 따로 출력
print("==============================val_loss======================================")
print(hist.history['val_loss']) #history에서 val_loss 값만 따로 출력
