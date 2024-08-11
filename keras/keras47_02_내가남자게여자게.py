import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
import pandas as pd


# path_submission = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'
# sample_submission=pd.read_csv(path_submission + "sample_submission.csv", index_col=0)

#1. data
np_path = 'c:/ai5/_data/image/me/'
x_test=np.load(np_path + 'me3.npy')


#2.modeling

# model=Sequential()
model=load_model('./_save/keras45/k45_07/k45_07_0805_1904_0037-0.1777.hdf5')

#3. compile

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])

start_time=time.time()

#4. 평가, 예측
# loss = model.evaluate(x_test, verbose=1)
# print('loss :', loss[0])
# print('acc :', round(loss[1],5))

# y_pre = np.round(model.predict(x_test, batch_size=16))

end_time=time.time()

print("걸린 시간 :", round(end_time-start_time,2),'초')

# print(y_pre)

### csv 파일 만들기 ###
y_submit = model.predict(x_test, batch_size=16)
# y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
print(y_submit)



end_time=time.time()
print("데이터 처리 걸린 시간 :", round(end_time-start_time,2),'초') #146.38초

x_train, x_test, y_train, y_test = train_test_split(x_train1, y_train1, train_size=0.9, random_state=1186)
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

