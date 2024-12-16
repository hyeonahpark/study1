import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,         # 이미지 스케일링
    horizontal_flip= True,  # 수평 뒤집기
    vertical_flip=True,     # 수집 뒤집기
    width_shift_range=0.1,  # 평행이동
    height_shift_range=0.1, # 평행이동 수직
    rotation_range=5,       # 각도 조절
    zoom_range=1.2,         # 축소 또는 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환    
    fill_mode='nearest',    # 이미지가 이동할 때 가장 가까운 곳의 색을 채운다는 뜻
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False, # 어지간하면 셔플할 필요 없음.
)  # Found 120 images belonging to 2 classes.

print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000001A310567760>
print(xy_train.next()) # dtype=float32), array([0., 0., 0., 1., 0., 1., 1., 1., 1., 0.], dtype=float32)) array 갯수 총 10개 = batch_size
print(xy_train.next()) # dtype=float32), array([0., 0., 1., 1., 1., 0., 0., 0., 1., 0.], dtype=float32)) 다 보고 싶으면 for문으로 돌리기~

print(xy_train[0]) 
print(xy_train[0][0]) # 1번째 x 데이터만 
print(xy_train[0][1]) # 1번째 y 데이터만
# print(xy_train[0].shape) # AttributeError: 'tuple' object has no attribute 'shape'
print(xy_train[0][0].shape) # (10, 200, 200, 1) [여기는 0~15][여기는 0~1]
# print(xy_train[16]) #ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[15][2]) #IndexError: tuple index out of range
print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>
