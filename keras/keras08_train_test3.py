import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn
from sklearn.model_selection import train_test_split

#1.data
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

# 슬라이싱 시 문제점 : 훈련된 데이터셋의 범위를 벗어난 지점부터는 loss가 점점 증가한다.
# 데이터셋에서 랜덤하게 추출하여 테스트시키면 loss가 증가하는 문제를 해결할 수 있다.

#[실습] 
# #[검색] train과 test를 섞어서 7:3으로 나눠라
# #힌트 : 사이킷런

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123) #shuffle defualt가 true, train_size 디폴트 0.75


#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=1)

#4. predict
print('x_train : ', x_train) # [ 6 9 4 2 7 10 3]
print('x_test : ', x_test) # [ 5 1 8]
print('y_train : ', y_train) # [6 9 4 2 7 10 3]
print('y_test : ', y_test) # [5 1 8]


print("+++++++++++++++++++++++++++++++++++++++")
loss=model.evaluate(x_test, y_test)
results=model.predict([11])
print("loss : ", loss) # 4.736951712906159e-15
print("[11]의 예측값 : ", results) #11.