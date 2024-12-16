# 배치를 160으로 잡고
# x,y를 추출해서 모델을 맹그러라
# acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time

start_time=time.time()
np_path = 'c:/ai5/_data/_save_npy/brain/'
x_train=np.load(np_path + 'keras44_01_x_train.npy')
y_train=np.load(np_path + 'keras44_01_y_train.npy')
x_test=np.load(np_path + 'keras44_01_x_test.npy')
y_test=np.load(np_path + 'keras44_01_y_test.npy')


end_time=time.time()
print("데이터 불러오는 시간 :", round(end_time-start_time,2),'초') #146.38초
# k44_01_0805_1157_0048-0.0012.hdf5

model=load_model('./_save/keras44/k44_01/k44_01_0805_1157_0048-0.0012.hdf5')
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