from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen =  ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=0.2,              # 축소 또는 확대
    shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

augment_size = 100
 
print(x_train.shape)  # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)


# plt.imshow(x_train[0], cmap='gray')
# plt.show()

aaa=np.tile(x_train[0], augment_size) #똑같은 타입을 100개 찍어서 reshape 
print(aaa.shape) # (100, 28, 28, 1)


xy_data = train_datagen.flow(
   np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1), # x
   np.zeros(augment_size), #y
   batch_size=augment_size,
   shuffle=False, 
)#.next() #.next를 한 경우에는 이터레이터 데이터 부분 앞에 배치([])를 가져옴, .next를 하지 않은 경우에는 이터레이터 데이터 형태기 때문에 [][]값을 가져옴 

# 튜플은 소괄호, 안에 바꿀 수 없음. 리스트는 대괄호, 안에 바꿀 수 있음.

print(xy_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x00000131A2CD6250>
print(type(xy_data)) # <class 'keras.preprocessing.image.NumpyArrayIterator'>

# print(xy_data.shape) #튜플은 shape 없음. list도 shape 없음!
print(len(xy_data)) # 1
# print(xy_data[0].shape) # AttributeError: 'tuple' object has no attribute 'shape'
print(xy_data[0][0].shape) # (100, 28, 28, 1)
print(xy_data[0][1].shape) # (100, )

plt.figure(figsize=(7,7))

for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][0][i], cmap='gray')
    
plt.show()
