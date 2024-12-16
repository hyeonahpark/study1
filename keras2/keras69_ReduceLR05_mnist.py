#m04_1에서 뽑은 4가지 결과로 4가지 모델을 맹그러
# input_shape=()
# 1. 70000,154
# 2. 70000,331
# 3. 70000,486
# 4. 70000,713
# 5. 70000,784

#시간과 성능 체크

#결과 예시
#결과 1. pca =154
#걸린시간 000초
# acc=000

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf 
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정
np.random.seed(337)
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

#1.data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x_train=x_train/255.
x_test=x_test/255.

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.transform(y_test.reshape(-1,1))

# print("0.95이상 : ",np.argmax(evr_cumsum>=0.95)+1) #154
# print("0.99이상 : ",np.argmax(evr_cumsum>=0.99)+1) #331
# print("0.999이상 : ",np.argmax(evr_cumsum>=0.999)+1) #486
# print("1.0 : ", np.argmax(evr_cumsum>=1.0)+1) #713

model= Sequential()
model.add(Dense(128, input_shape=(784, )))
model.add(Dense(256,  activation = 'relu'))
model.add(Dense(512,  activation = 'relu'))
model.add(Dense(1024,  activation = 'relu'))
model.add(Dense(512,  activation = 'relu'))
model.add(Dense(256,  activation = 'relu'))
model.add(Dense(128,  activation = 'relu'))
model.add(Dense(10, activation='softmax'))


#3. compile
from tensorflow.keras.optimizers import Adam
learning_rate=0.0001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=0, factor=0.8) #factor는 곱하기!

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")



path = 'C:\\ai5\\_save\\keras69\\k69_05\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k69_05_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose =1, validation_split=0.2, callbacks=[es, mcp, rlr])
end_time=time.time()


#4. predict

loss=model.evaluate(x_test, y_test, verbose=1)
y_test1 = np.argmax(y_test, axis=1)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)

print("결과.lr :", learning_rate)
print("loss : ", loss[0])
print("ACC : ", round(loss[1], 6))
print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# 결과.lr : 0.1
# loss :  2.3111448287963867
# ACC :  0.0982
# 걸린 시간 :  38.35 초

# 결과.lr : 0.01
# loss :  2.3018500804901123
# ACC :  0.101
# 걸린 시간 :  43.71 초

# 결과.lr : 0.005
# loss :  0.3022957146167755
# ACC :  0.9309
# 걸린 시간 :  39.47 초

# 결과.lr : 0.001
# loss :  0.10615439713001251
# ACC :  0.9744
# 걸린 시간 :  71.31 초

# 결과.lr : 0.0005
# loss :  0.09253744035959244
# ACC :  0.9771
# 걸린 시간 :  64.95 초

# 결과.lr : 0.0001
# loss :  0.08773382753133774
# ACC :  0.9776
# 걸린 시간 :  81.54 초



###################################
# 결과.lr : 0.0001
# loss :  0.08819566667079926
# ACC :  0.9736
# 걸린 시간 :  125.61 초
###################################