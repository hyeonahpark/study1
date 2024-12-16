import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. data
x=np.array(range(1, 17))
y=np.array(range(1, 17))

#[실습] 잘라라!
#train_test_split로만 잘라라

# x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.75, random_state=104)
# x_train, x_val, y_train, y_val= train_test_split(x, y, train_size=0.75, random_state=104)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.65, random_state=133)

print(x_train, y_train)
print(x_test, y_train)

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.3)#(x_val, y_val))  #x_train, y_train 값을 7:3으로 나눔

#4. predict
print("+++++++++++++++++++++++++++++++++++++++")
loss=model.evaluate(x_test, y_test) #verbose=0) #evaluate 에도 verbose가 존재함.
results=model.predict([18])
print("loss : ", loss)
print("[18]의 예측값 : ", results)