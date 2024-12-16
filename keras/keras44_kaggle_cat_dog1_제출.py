import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
import pandas as pd


path_submission = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'
sample_submission=pd.read_csv(path_submission + "sample_submission.csv", index_col=0)

#1. data
np_path = 'c:/ai5/_data/_save_npy/kaggle_cat_dog/'

x_train1=np.load(np_path + 'keras43_02_x_train1.npy')
y_train1=np.load(np_path + 'keras43_02_y_train1.npy')
x_train2=np.load(np_path + 'keras43_02_x_train2.npy')
y_train2=np.load(np_path + 'keras43_02_y_train2.npy')
x_test=np.load(np_path + 'keras43_02_x_test.npy')
y_test=np.load(np_path + 'keras43_02_y_test.npy')


#2.modeling
# model=Sequential()


#3. compile

model=load_model('./_save/keras42/k42_01/k42_01_0804_2356_0059-0.2163.hdf5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','acc', 'mse'])

start_time=time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = np.round(model.predict(x_test, batch_size=16))

end_time=time.time()


print("걸린 시간 :", round(end_time-start_time,2),'초')

"""
### csv 파일 만들기 ###
y_submit = model.predict(x_test, batch_size=16)
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
# print(y_submit)

# print(y_submit)
sample_submission['label'] = y_submit
sample_submission.to_csv(path_submission + "teacher0805_2.csv")
"""