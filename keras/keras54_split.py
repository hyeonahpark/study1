import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1,11))
# print(a) #[ 1  2  3  4  5  6  7  8  9 10]
size = 4


def split_x(dataset, size) :
    aaa=[]
    for i in range(len(dataset)-size+1):
        subset = dataset[i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)

x=bbb[:, :-1]
y=bbb[:, -1]
# print(x)
print(x.shape, y.shape)  #(7, 3) (7,)
x = x.reshape(7, 3, 1)


#2. MODELING
model=Sequential()
# model.add(SimpleRNN(10, input_shape=(3,1))) #데이터의 갯수 7을 빼고 나머지를 shape에 넣어줌, #3 : timesteps, features
model.add(LSTM(15, input_shape=(3,1))) #LSTM 사용
# model.add(GRU(10, input_shape=(3,1))) #GRU 사용
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(1))
#DNN 에서는 units, CNN에서는 filters, RNN에서는 units
#LSTM은 적은 데이터로는 성능이 크게 좋지 않지만, 데이터 양이 많을수록 성능이 좋아짐. BUT, 속도는 느려! 


#3.compile
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=15000)

#4.predict
results=model.evaluate(x,y)
print('loss:', results)

x_pred=np.array([8,9,10]).reshape(1,3,1) #(1, 4, 1)
y_pred = model.predict(x_pred)
print('[8,9,10]의 결과 : ', y_pred) 


#loss: 2.2481647192762466e-06
#[8,9,10]의 결과 :  [[10.957633]]