#18-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
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



#2. modeling
model=Sequential()
model.add(Dense(10, input_dim=13)) # 특성은 항상 많으면 좋음! 데이터가 많으면 좋으니까
# model.add(Dense(10, input_shape=(13, ))) # 이미지 input_shape=(8,8,1)
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(1))

# 한계 : 항상 input_dim은 행렬 형태였음. 하지만 다차원 행렬의 경우에는 input.shape를 해줘야 함.
# 이미지에서 8*8이 100개 있는 경우 100*8*8 => 100*64로 변경 가능 / 컬러인 경우는 100*8*8*3 (RGB)


#3. compile
model.compile(loss='mse', optimizer='adam')
start_time=time.time() #time.time() 현재 시간 반환
hist = model.fit(x_train, y_train, epochs=1000, batch_size=4, verbose = 1, validation_split=0.2) #hist는 history의 약자,
end_time=time.time() #끝나는 시간 반환

#4. predict

loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)
print("걸린 시간 : ", round(end_time - start_time, 2), "초") #round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


#loss :  21.243154525756836, R2의 점수 :  0.7304632002256847 (random_state=54, test_size=0.25, epochs=500)
#loss :  20.018720626831055, R2의 점수 :  0.7432149994897693 (random_state=54, test_size=0.3, epochs=500)
#loss :  19.934724807739258, R2의 점수 :  0.752942583325209 (random_state=54, test_size=0.27, epochs=1000)
#loss :  17.43680763244629, R2의 점수 :  0.7510706438773507 (random_state=55, test_size=0.27, epochs=10000)
#loss :  17.58778953552246, R2의 점수 :  0.7552544152244796 (random_state=55, test_size=0.26, epochs=10000)
#loss :  16.94853973388672, R2의 점수 :  0.7508356644979253 (random_state=55, test_size=0.25, epochs=10000)
#loss :  21.8568058013916, R2의 점수 :  0.7632840808134838 (random_state=333, test_size=0.25, epochs=10000)
#loss :  25.599809646606445, R2의 점수 :  0.764170205161022 (random_state=6666, tese_size=0.2, epochs=1000)
#loss :  24.135040283203125, R2의 점수 :  0.7776639093439579 (random_state=6666, tese_size=0.2, epochs=1000) 13 10 10 5 3 1 1
#loss :  23.845094680786133, R2의 점수 :  0.7803349286808787 (random_state=6666, tese_size=0.2, epochs=1000) 13 10 7 5 3 1 1
#loss :  23.40278434753418, R2의 점수 :  0.7844095751777879
#loss :  23.67738914489746, R2의 점수 :  0.7818798717880635 (random_state=6666, tese_size=0.2, epochs=1000) 13 7 3 3 1 1

#============================================================================================================== minmaxscaler
#loss :  20.878190994262695, R2의 점수 :  0.8076665579823633, 걸린 시간 :  15.96 초
#loss :  21.558116912841797, R2의 점수 :  0.8014029495461564, 걸린 시간 :  46.67 초

#============================================================================================================== standardscaler
#loss :  23.681394577026367, R2의 점수 :  0.7818429710326453
#loss :  22.218841552734375. R2의 점수 :  0.79531624308922

#=======================================================maxabs
#loss :  22.177705764770508 ,R2의 점수 :  0.7956952005957241

#=======================================================robust
#loss :  22.547109603881836, R2의 점수 :  0.7922922098150922