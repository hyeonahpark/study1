from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

x,y = load_digits(return_X_y=True) #return_X_y= True로 작성 가능

print(x)
print(y)
print(x.shape, y.shape) #(1797, 64) (1797,) 이미지면 (1797,8,8)

print(pd.value_counts(y,sort=False)) #sort=False를 하면 순서대로 정렬됨
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

y=pd.get_dummies(y)
print(y.shape) #(1797, 10)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. modeling
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim=64))
model.add(Dense(200,  activation = 'relu'))
model.add(Dense(300,  activation = 'relu'))
model.add(Dense(600,  activation = 'relu'))
model.add(Dense(300,  activation = 'relu'))
model.add(Dense(200,  activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation='softmax'))


#3. compile

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])
start_time=time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode='min', patience=100, restore_best_weights=True)


model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, validation_split=0.3, callbacks=[es])
end_time=time.time()

#4.predict

loss=model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
print(y_predict)

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수

#======================================
#loss :  0.19780610501766205, ACC :  0.978
#========================================
#loss :  0.2500123083591461, ACC :  0.944