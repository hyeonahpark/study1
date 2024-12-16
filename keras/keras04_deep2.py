from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #수치데이터 때문에 사용

#1. data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss를 만들어라
#epochs 100으로 고정
#로스 기준 0.33 미만

#2. modeling
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(1))


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


# 로스: 0.32450738549232483
# 로스: 0.3242216408252716
# 로스: 0.32381144165992737
# 로스: 0.3238101303577423