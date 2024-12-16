import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn
from sklearn.model_selection import train_test_split

#09_1 copy
#R2 검색

#1.data
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=104)

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)  #loss :  4.290445804595947

y_predict=model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error

r2= r2_score(y_test, y_predict)

print("R2의 점수 : ", r2) #R2의 점수 :  0.8851593719725027

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
