from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, LSTM
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

x=x.reshape(1797,64,1)
x=x/255.


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186, stratify=y)

#2. modeling
from keras.layers import Dropout
model=Sequential()
model.add(LSTM(64, input_shape=(64, 1),return_sequences=True)) 
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True)) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
model.add(LSTM(32)) 
# model.add(MaxPool2D())
model.add(Dropout(0.25))
# model.add(Conv2D(32, (3,3), activation='relu', padding='same')) 
# model.add(MaxPool2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  #23*23*32
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, input_shape=(32, ), activation='relu')) 
                        #shpae = (batch_size, input_dim)
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))



#3. compile

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras59\\k59_11\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k59_11_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras39/k39_11/keras39_11_mcp.hdf5')

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

# loss :  0.16577006876468658
# ACC :  0.956

#dropout
# loss :  0.16080109775066376
# ACC :  0.972

# loss :  0.13260766863822937
# ACC :  0.961

#loss :  0.07087568938732147
# ACC :  0.983

#LSTM
# loss :  0.7677649855613708
# ACC :  0.733
# 걸린 시간 :  20.92 초