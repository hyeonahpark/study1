import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten
import time
from sklearn.model_selection import train_test_split

#1. data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train) #다 0이 나오는 이유는 특성을 가진 값은 가운데에 몰려있기 때문
# print(x_train[0])
# print("y_train[0] : ", y_train[0]) #5

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> (60000,28,28,1) 과 동일. 데이터의 값과 순서의 변화가 없기 때문
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

#x를 3차원->4차원 데이터로 만들기 reshape

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#y는 OneHot Encoding 해서 60000,10으로 만들어주기

y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)

print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000, 10)
print(x_test.shape, y_test.shape) #(10000, 28, 28, 1) (10000, 10)

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y)

#2. modeling
model = Sequential()
model.add(Conv2D(10, (3,3), input_shape=(28, 28, 1))) #27, 27, 10
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임

model.add(Conv2D(filters=20, kernel_size=(3,3))) # 25, 25, 20
model.add(Conv2D(15, (4,4))) # 22, 22, 15
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0

model.add(Dense(units=8)) #None, 22, 22, 8 #Dense가 2차원이지만 2차원 이상 다 가능함
model.add(Dense(units=9, input_shape=(8,))) #22, 22, 9
                        #shpae = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))
                        
model.summary()
                        
#param 수 = (커널 수*채널 수 + bias(1))*filter 수
                        
#3. compile

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=10,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time()
model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=1, validation_split=0.1, callbacks=[es])
end_time=time.time()


#4. predict

loss=model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
print(y_predict)

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

# CPU
# loss :  0.3052709698677063
# ACC :  0.918
# 걸린 시간 :  495.78 초

#GPU
#loss :  0.29991039633750916
# ACC :  0.921
# 걸린 시간 :  57.53 초