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
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 이미지 스케일링
    horizontal_flip= True,  # 수평 뒤집기
    vertical_flip=True,     # 수집 뒤집기
    width_shift_range=0.1,  # 평행이동
    height_shift_range=0.1, # 평행이동 수직
    rotation_range=1,       # 각도 조절
    zoom_range=0.2,         # 축소 또는 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환    
    fill_mode='nearest',    # 이미지가 이동할 때 가장 가까운 곳의 색을 채운다는 뜻
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\train\\'
path_test = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\test\\'
path_submission = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'
sample_submission=pd.read_csv(path_submission + "sample_submission.csv", index_col=0)

xy_train1 = train_datagen.flow_from_directory(
    path_train, 
    target_size=(100,100),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 25000 images belonging to 2 classes.

xy_train2 = test_datagen.flow_from_directory(
    path_train,            
    target_size=(100,100),  
    batch_size=25000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(100,100),
    batch_size=12500,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False, # 어지간하면 셔플할 필요 없음.
)  # Found 12500 images belonging to 1 classes.

x = np.concatenate((xy_train1[0][0][:6000],xy_train2[0][0]))
y = np.concatenate((xy_train1[0][1][:6000],xy_train2[0][1]))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

# x_train=xy_train[0][0]
# y_train=xy_train[0][1]
# x_test=xy_test[0][0]
# y_test=xy_test[0][1]

# print(xy_train[0][0].shape)  #(25000, 100, 100, 3)
# print(xy_train[0][1].shape)  #(25000, )


np_path = 'c:/ai5/_data/_save_npy/kaggle_cat_dog/'
np.save(np_path + 'keras43_02_x_train1.npy', arr=xy_train1[0][0])
np.save(np_path + 'keras43_02_y_train1.npy', arr=xy_train1[0][1])
np.save(np_path + 'keras43_02_x_train2.npy', arr=xy_train2[0][0])
np.save(np_path + 'keras43_02_y_train2.npy', arr=xy_train2[0][1])
np.save(np_path + 'keras43_02_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras43_02_y_test.npy', arr=xy_test[0][1])



end_time=time.time()
print("데이터 처리 걸린 시간 :", round(end_time-start_time,2),'초') #146.38초

# xy_test=xy_test[0][0]