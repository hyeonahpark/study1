import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)
from tensorflow.keras.layers import Flatten, Dropout
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#### 스케일링 1-1 ######
x_train = x_train/255.
x_test = x_test/255.

#2. modeling
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))) #26, 26, 64
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3))) # 24, 24, 64
model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2), activation='relu')) # 23, 23, 32
model.add(Dropout(0.3))
#model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  #23*23*32
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu')) #None, 22, 22, 8 #Dense가 2차원이지만 2차원 이상 다 가능함
model.add(Dense(units=16, input_shape=(32,), activation='relu')) 
                        #shpae = (batch_size, input_dim)
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
                        
# model.summary()
                        
                        
#3. compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras35\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k35_05_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=3000, batch_size=128, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

# model.save('./_save/keras35/keras35_04_mcp.hdf5')

#4. predict

loss=model.evaluate(x_test, y_test)
#y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
print(y_predict)


from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# loss :  0.2710987627506256
# ACC :  0.91

# loss :  0.28665873408317566
# ACC :  0.904
# 걸린 시간 :  987.89 초