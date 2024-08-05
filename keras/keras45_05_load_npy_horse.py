
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

start_time=time.time()
np_path = 'c:/ai5/_data/_save_npy/horse/'
x_train=np.load(np_path + 'keras44_02_x_train.npy')
y_train=np.load(np_path + 'keras44_02_y_train.npy')


end_time=time.time()
print("데이터 불러오는 시간 :", round(end_time-start_time,2),'초') #146.38초

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=5656)

model=load_model('./_save/keras44/k44_02/k41_04_0802_1502_0021-0.0023.hdf5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])

start_time=time.time()

#4. 평가, 예측
from sklearn.metrics import r2_score, accuracy_score


loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1], 5))

y_pre = np.round(model.predict(x_test))
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end_time-start_time,2),'초')
r2=r2_score(y_test, y_pre)
print("R2의 점수 : ", r2)