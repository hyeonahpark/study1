import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. data
x=np.array(range(1, 17))
y=np.array(range(1, 17))

#[실습] data 자르기

x_train=x[:10] 
y_train=y[:10]

x_val=x[10:13]
y_val=y[10:13]

x_test=x[13:]
y_test=x[13:]

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state = 123)

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_data=(x_val, y_val))  

#4. predict
print("+++++++++++++++++++++++++++++++++++++++")
loss=model.evaluate(x_test, y_test)
results=model.predict([17])
print("loss : ", loss)
print("[17]의 예측값 : ", results)