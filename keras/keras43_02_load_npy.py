#https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/leaderboard

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import os

#1. data (시간체크)
start_time=time.time()


np_path = 'c:/ai5/_data/_save_npy/kaggle_cat_dog/'
# np.save(np_path + 'keras43_01_x_train1.npy', arr=xy_train1[0][0])
# np.save(np_path + 'keras43_01_y_train1.npy', arr=xy_train1[0][1])
# np.save(np_path + 'keras43_01_x_train2.npy', arr=xy_train2[0][0])
# np.save(np_path + 'keras43_01_y_train2.npy', arr=xy_train2[0][1])
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

x_train1=np.load(np_path + 'keras43_01_x_train1.npy')
y_train1=np.load(np_path + 'keras43_01_y_train1.npy')
x_train2=np.load(np_path + 'keras43_01_x_train2.npy')
y_train2=np.load(np_path + 'keras43_01_y_train2.npy')
x_test=np.load(np_path + 'keras43_01_x_test.npy')
y_test=np.load(np_path + 'keras43_01_y_test.npy')

print(x_train1)
print(x_train1.shape) #(25000, 100, 100, 3)
print(y_train1) 
print(y_train1.shape) #(25000,)
print(x_train2.shape) #(25000, 100, 100, 3)
print(y_train2.shape) #(25000,)
print(x_test.shape) #(12500, 100, 100, 3)
print(y_test.shape) #(12500,) 


end_time=time.time()
print("데이터 불러오는 시간 :", round(end_time-start_time,2),'초') #146.38초

# xy_test=xy_test[0][0]