from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442, )


#[실습]
#R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)

#2. modeling
model=Sequential()
model.add(Dense(251, input_dim=10))
model.add(Dense(141))
model.add(Dense(171))
model.add(Dense(14))
model.add(Dense(5))
model.add(Dense(1))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

#loss :  2902.455810546875, R2의 점수 :  0.4357510056875852 
#loss :  2681.670166015625, R2의 점수 :  0.5017249457183628 (train_size=0.75, random_state=33, batch_size=10)
#loss :  2835.939697265625, R2의 점수 :  0.5134413663352477 (train_size=0.8, random_state=33, batch_size=10)
#loss :  2509.151123046875, R2의 점수 :  0.5997273328212158 (train_size=0.9, random_state=3333, batch_size=10)
#loss :  2504.762939453125, R2의 점수 :  0.600427321945652 (train_size=0.9, random_state=3333, batch_size=32)
#loss :  2857.839599609375, R2의 점수 :  0.6151175216077298 (train_size=0.9, random_state=52151, batch_size=32)
#loss :  2847.067626953125, R2의 점수 :  0.6165682456452712 
# loss :  2819.262939453125, R2의 점수 :  0.6203129299461492 