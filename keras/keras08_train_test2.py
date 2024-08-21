import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.data
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱 !! 7:3으로 잘라라 !!

x_train=x[:7] 
y_train=y[:7]

x_test=x[7:]
y_test=x[7:]

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=1)

#4. predict
print("+++++++++++++++++++++++++++++++++++++++")
loss=model.evaluate(x_test, y_test)
results=model.predict([11])
print("loss : ", loss)
print("[11]의 예측값 : ", results)


