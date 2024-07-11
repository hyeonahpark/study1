from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #수치데이터 때문에 사용

#1. data
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. modeling
model=Sequential()
model.add(Dense(1, input_dim=1))

epochs = 100000
#3. compile, traning
model.compile(loss= 'mse', optimizer='adam')
model.fit(x,y, epochs=epochs)

#4. result, predict
loss=model.evaluate(x,y)
print("=============================================")
print("epochs : ", epochs)
print("로스:", loss) 
result=model.predict(([6]))
print("6의 예측값: ", result)

#6의 예측값:  [[5.7000008]], 로스 : 0.37999996542930603, epochs=60000
#6의 예측값:  [[5.7000113]], 로스 : 0.37999996542930603, epochs=50000
#6의 예측값:  [[5.7000194]] , 로스: 0.3800000548362732, epochs=30000
#6의 예측값:  [[5.700001]] , 로스: 0.37999996542930603, epochs=20000
#6의 예측값:  [[5.6734104]] , 로스: 0.38025689125061035, epochs=10000
#6의 예측값:  [[5.700004]] , 로스: 0.37999996542930603, epochs=5000
#6의 예측값:  [[4.1922054]], 로스: 0.8953554034233093, epochs=1000
#6의 예측값:  [[-0.27483734]], 로스: 11.76978874206543, epochs=100