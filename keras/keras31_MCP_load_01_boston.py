import numpy as np
from tensorflow.keras.models import Sequential, load_model #load_model : model 을 불러옴
from tensorflow.keras.layers import Dense
import sklearn as sk
print(sk.__version__) #0.24.2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time

#1.data
dataset=load_boston()

x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=6666)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train)) #0.0 1.0000000000000002 #부동소수점 연산이기 때문에 단순한 연산오류임. 원래는 1.0나와야하눈디 ! !
print(np.min(x_test), np.max(x_test)) #-0.008298755186722073 1.1478180091225068


print("===================mcp 출력 =========================")
model= load_model('./_save/keras30_mcp/k30_1/keras30_1_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test, verbose = 0)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)


#loss :  22.382080078125
# R2의 점수 :  0.793812460823792
# 걸린 시간 :  2.03 초

#loss :  22.382080078125
# R2의 점수 :  0.793812460823792
