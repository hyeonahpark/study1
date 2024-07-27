#26-1 copy

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
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']


# print(dataset)
x=dataset.data
y=dataset.target


# print(x)
# print(x.shape) #(506,13)
# print(y)
# print(y.shape) #(506, )

#train_size : 0.7~0.9 사이로
#R2 0.8 이상
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



# #2. modeling
model=Sequential()
model.add(Dense(10, input_dim=13)) # 특성은 항상 많으면 좋음! 데이터가 많으면 좋으니까
# model.add(Dense(10, input_shape=(13, ))) # 이미지 input_shape=(8,8,1)
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(1))

# model.summary()

# model.load_weights("C:\\ai5\\study\\_save\\keras28\\keras28_1_save_model.h5") #모델만 저장 

#load_weights 는 원 모델 붙여줘야하고, 컴파일도 해줘야함, 파일 용량 차이가 있음

# model=load_model("C:\\ai5\\study\\_save\\keras28\\keras28_5_save_weights2.h5")
model.load_weights("./_save/keras28/keras28_5_save_weights1.h5")

model.summary()


# #3. compile
model.compile(loss='mse', optimizer='adam') #load_weights 할때는 항상 컴파일 넣기
# start_time=time.time() #time.time() 현재 시간 반환
# hist = model.fit(x_train, y_train, epochs=10, batch_size=4, verbose = 1, validation_split=0.2) #hist는 history의 약자,
# end_time=time.time() #끝나는 시간 반환


# model.save("C:\\ai5\\study\\_save\\keras28\\keras28_1_save_mode2.h5") #모델만 저장


#4. predict

loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)
# print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#save model : 모델, 가중치 저장 가능 / #2 다음에 save_model 하면 모델만 저장, #3 다음에 save_model 하면 가중치도 같이 저장 가능 -> 그러면 계속 같은 결과 호출됨
