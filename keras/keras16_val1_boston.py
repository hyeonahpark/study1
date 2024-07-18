import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
print(sk.__version__) #0.24.2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

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
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, random_state=6666)

#2. modeling
model=Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=32, verbose=0, validation_split=0.2)

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

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

#갱신~!
#loss :  20.38176155090332, R2의 점수 :  0.8361604175711215
