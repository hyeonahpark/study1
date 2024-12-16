# 배치를 160으로 잡고
# x,y를 추출해서 모델을 맹그러라
# acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time

train_datagen = ImageDataGenerator(
    rescale=1./255,         # 이미지 스케일링
    # horizontal_flip= True,  # 수평 뒤집기
    # vertical_flip=True,     # 수집 뒤집기
    # width_shift_range=0.1,  # 평행이동
    # height_shift_range=0.1, # 평행이동 수직
    # rotation_range=5,       # 각도 조절
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환    
    # fill_mode='nearest',    # 이미지가 이동할 때 가장 가까운 곳의 색을 채운다는 뜻
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(200,200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(200,200),
    batch_size=120,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False, # 어지간하면 셔플할 필요 없음.
)  # Found 120 images belonging to 2 classes.

np_path = 'c:/ai5/_data/_save_npy/brain/'
np.save(np_path + 'keras44_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras44_01_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras44_01_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras44_01_y_test.npy', arr=xy_test[0][1])

# 