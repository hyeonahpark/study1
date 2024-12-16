import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn
from sklearn.model_selection import train_test_split

#1.data
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=64)

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=1)

#4. predict
print('x_train : ', x_train) # [20 13  2  1  3 15 18 11  9 16 17 19  7  5]
print('x_test : ', x_test) # [10  6  4 12  8 14]
print('y_train : ', y_train) # [20 14  2  1  4  9 23 13  8  6 17 21  9  5]
print('y_test : ', y_test) # [12  7  3  8  3 15]


print("+++++++++++++++++++++++++++++++++++++++")
loss=model.evaluate(x_test, y_test)
results=model.predict([x])
print("loss : ", loss)  # 7.695509433746338
print("[x]의 예측값 : ", results) # [[ 1.3705189]
#  [ 2.3085907]
#  [ 3.2466621]
#  [ 4.184734 ]
#  [ 5.1228056]
#  [ 6.0608773]
#  [ 6.998949 ]
#  [ 7.937021 ]
#  [ 8.8750925]
#  [ 9.813164 ]
#  [10.751236 ]
#  [11.689307 ]
#  [12.627379 ]
#  [13.565451 ]
#  [14.503523 ]
#  [15.441594 ]
#  [16.379665 ]
#  [17.317738 ]
#  [18.25581  ]
#  [19.19388  ]]


import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x, results, color='red')
plt.show()