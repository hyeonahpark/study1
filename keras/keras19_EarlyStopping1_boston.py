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
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(1))


#3. compile
model.compile(loss='mse', optimizer='adam')
start_time=time.time() #time.time() 현재 시간 반환

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=10,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose = 1, validation_split=0.2, callbacks=[es]) #hist는 history의 약자,
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


import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) #그림판 사이즈 설정
plt.plot(hist.history['loss'], c='red', label='loss',)
plt.plot(hist.history['val_loss'], c='blue', label='val_loss',)
plt.legend(loc='upper right') #범례, 범례 위치 위에 오른쪽
plt.title('Boston Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

# plt.plot(hist.history['loss'], c='red', label='loss', marker='.')

plt.show()

# dict : immutable한 키(key)와 mutable한 값(value)으로 맵핑되어 있는 순서가 없는 집합