from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D #이미지 작업 Convolution2D == Conv2D (1D : 선, 3D : 입체형태)

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape =(5, 5, 1))) # 2,2 로 자름(커널사이즈) / 5: 가로, 5: 세로, 1: 컬럼수 / 몇 장인지는 모름 (행무시) => 이미지 처리 작업을 거치면 (4,4,10)이 됨 -> 데이터 압축 후 증폭
model.add(Conv2D(5, (2,2))) # 2,2 로 자름(커널사이즈) / 5: 가로, 5: 세로, 1: 컬럼수 / 몇 장인지는 모름 (행무시) => 이미지 처리 작업을 거치면 (4,4,10)이 됨 -> 데이터 압축 후 증폭

model.summary()

#_________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50        

#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0
# _________________________________________________________________

