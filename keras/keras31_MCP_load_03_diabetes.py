from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442, )


#[실습]
#R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


print("===================mcp 출력 =========================")
model= load_model('./_save/keras30_mcp/k30_3/keras30_3_mcp.hdf5')

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

#loss :  2887.905517578125
# R2의 점수 :  0.6110683426680202

#loss :  2887.905517578125
# R2의 점수 :  0.6110683426680202