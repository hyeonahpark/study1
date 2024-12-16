from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,4,5,6])

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile, traning
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=300)

#4. predict
loss=model.evaluate(x, y)
print("로스: ", loss) #로스 출력줄 추가
result=model.predict([1,2,3,4,5,6,7])
print("7의 예측값 :", result)