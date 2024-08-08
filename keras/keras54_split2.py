import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]])

a=a.T

# print(a)
# print(len(a))
# print(a.shape) # (10, 2)

size = 6


def split_x(dataset, size) :
    aaa=[]
    for i in range(len(dataset)-size+1):
        subset = dataset[i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x(a, size)
# print(bbb) #[[[ 1  9][ 2  8][ 3  7][ 4  6][ 5  5][ 6  4]], 
# #  [[ 2  8][ 3  7][ 4  6][ 5  5][ 6  4][ 7  3]],
# #  [[ 3  7][ 4  6][ 5  5] [ 6  4][ 7  3] [ 8  2]]
# #  [[ 4  6][ 5  5][ 6  4][ 7  3][ 8  2][ 9  1]]
# #  [[ 5  5] [ 6  4][ 7  3][ 8  2][ 9  1][10  0]]]
# print(bbb.shape) #(5,6,2)



x=bbb[:, :-1]
y=bbb[:, -1, 0]
# print(x)
# [[[1 9][2 8][3 7][4 6][5 5]]
# [[2 8][3 7][4 6][5 5][6 4]]
# [[3 7] [4 6][5 5][6 4][7 3]]
# [[4 6][5 5][6 4] [7 3][8 2]]
# [[5 5][6 4] [7 3][8 2] [9 1]]]
print(y)
# [ 6  7  8  9 10]
print(x.shape, y.shape) # (5,5,2) (5,2)


#2. MODELING
model = Sequential()
model.add(LSTM(units=32, input_shape=(5,2), return_sequences=True)) # timesteps , features
model.add(LSTM(32, return_sequences=True)) # timesteps , features
model.add(LSTM(32))
# Flaten 사용하는 방법도 있음 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#DNN 에서는 units, CNN에서는 filters, RNN에서는 units
#LSTM은 적은 데이터로는 성능이 크게 좋지 않지만, 데이터 양이 많을수록 성능이 좋아짐. BUT, 속도는 느려! 


#3.compile
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=4, verbose=3)

#4.predict
results=model.evaluate(x,y)
print('loss:', results)

x_pred=np.array([[6,4],[7,3],[8,2],[9,1],[10,0]]).reshape(1,5,2)
y_pred = model.predict(x_pred)
print('[6,4],[7,3],[8,2],[9,1],[10,0]의 결과 : ', y_pred) 


#loss: 2.2481647192762466e-06
#[8,9,10]의 결과 :  [[10.957633]]

