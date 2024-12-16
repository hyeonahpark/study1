from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #수치데이터 때문에 사용

#1. data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 예측하라
#epochs 100으로 고정
#소수 네째자리까지 맞추면 합격. 예: 6.000 또는 5.9999

#2. modeling
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(3, input_dim=3))
model.add(Dense(5, input_dim=3))
# model.add(Dense(11, input_dim=9))
model.add(Dense(9, input_dim=5))
model.add(Dense(5, input_dim=9))
model.add(Dense(3, input_dim=5))
model.add(Dense(1, input_dim=3))
# model.add(Dense(1, input_dim=3))

epochs = 100
#3. compile, traning
model.compile(loss= 'mse', optimizer='adam')
model.fit(x,y, epochs=epochs)

#4. result, predict
loss=model.evaluate(x,y)
print("=============================================")
print("epochs : ", epochs)
print("로스:", loss) 
result=model.predict(([6]))
print("6의 예측값: ", result)


# 6의 예측값:  [[6.001249]] 1 3 9 3 1
# 6의 예측값:  [[5.999841]] 1 3 5 9 1