import numpy as np
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
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

#2. modeling
model=Sequential()
model.add(Dense(3, input_dim=8))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile
model.compile(loss='mse', optimizer='adam')
start_time=time.time()
hist=model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3)
end_time=time.time()

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

#loss :  0.6239640712738037, R2의 점수 :  0.5316343085808204 (random_state=54, test_size=0.25)
#loss :  0.616243302822113, R2의 점수 :  0.5371893365153388 (random_state=666, test_size=0.25)
#loss :  0.6237843036651611, R2의 점수 :  0.5396757389363518 (random_state=1004, test_size=0.25)
#loss :  0.5586398839950562, R2의 점수 :  0.5842076076551155 (random_state=3, test_size=0.13)
#loss :  0.5393274426460266, R2의 점수 :  0.5997427075615596 (random_state=3, test_size=0.1)

#====================================================================minmaxscaler
#loss :  0.5018455982208252, R2의 점수 :  0.627559618950168

#====================================================================standardscaler
#loss :  0.49793171882629395, R2의 점수 :  0.6304642637516715
#loss :  0.49745675921440125, R2의 점수 :  0.6308167215875821

#=====================maxabs
#loss :  0.5503730177879333, R2의 점수 :  0.5915453315733987

#=====================robust
#loss :  0.49917295575141907, R2의 점수 :  0.6295431247649544