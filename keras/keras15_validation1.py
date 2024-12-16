# 08-1 copy
#이 파일에서 validation_data=(x_val, y_val) 만 추가
#데이터 3등분 (train, validation, test)
#통상적으로 loss 보다 val_loss가 더 안좋고, evaluate 한게 val_loss보다 더 안좋음.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
# x=np.array([1,2,3,4,5,6,7,8,9,10])
# y=np.array([1,2,3,4,5,6,7,8,9,10])

x_train=np.array([1,2,3,4,5,6])
y_train=np.array([1,2,3,4,5,6])

x_val = np. array ([7,8]) 
y_val = np. array ([7,8])

x_test=np.array([9,10])
y_test=np.array([9,10])

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_data=(x_val, y_val))  
#verbose = 1 값이 디폴트임. 
#verbose = 0으로 설정할 경우 컴파일 과정이 보이지 않고 바로 값 출력, verbose = 2로 설정할 경우 진행바가 나오지 않음

#verbose = 0 : 침묵
#verbose = 1 : 디폴트 
#verbose = 2 : 프로그래스바 삭제
#verbose = 나머지 : 에포만 나온다.

#4. predict
print("+++++++++++++++++++++++++++++++++++++++")
loss=model.evaluate(x_test, y_test)
results=model.predict([11])
print("loss : ", loss)
print("[11]의 예측값 : ", results)