import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import time

#1. data
x=np.array(range(1, 17))
y=np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.65, random_state=133)

print(x_train, y_train)
print(x_test, y_train)

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
start_time=time.time() #time.time() 현재 시간 반환
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.3)#(x_val, y_val))  #x_train, y_train 값을 7:3으로 나눔
end_time=time.time() #끝나는 시간 반환

#4. predict
print("+++++++++++++++++++++++++++++++++++++++")
loss=model.evaluate(x_test, y_test) #verbose=0) #evaluate 에도 verbose가 존재함.
results=model.predict([18])
print("loss : ", loss)
print("[18]의 예측값 : ", results)

print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수
