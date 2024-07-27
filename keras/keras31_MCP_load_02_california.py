import numpy as np
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. data
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(20640, 8) (20640, )

#[실습]
#R2 0.59 이상

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=3, )


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler=RobustScaler()
# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


print("===================mcp 출력 =========================")
model= load_model('./_save/keras30_mcp/k30_2/keras30_2_mcp.hdf5')


#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

# loss :  0.49470725655555725
# R2의 점수 :  0.6328572745535846

# loss :  0.49470725655555725
# R2의 점수 :  0.6328572745535846
