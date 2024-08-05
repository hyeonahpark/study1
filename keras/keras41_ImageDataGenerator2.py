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

x_train=xy_train[0][0]
y_train=xy_train[0][1]
x_test=xy_test[0][0]
y_test=xy_test[0][1]


#2. modeling
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1), padding='same')) #26, 26, 64
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
model.add(MaxPool2D())
model.add(Conv2D(filters=32, activation='relu', kernel_size=(3,3), padding='same')) # 24, 24, 64
model.add(MaxPool2D())
model.add(Conv2D(filters=32, activation='relu', kernel_size=(3,3), padding='same')) # 24, 24, 64
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) # 23, 23, 32
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Flatten()) # 모양만 바꾼거기 때문에 연산량 0  #23*23*32
model.add(Dense(units=32, activation='relu')) #None, 22, 22, 8 #Dense가 2차원이지만 2차원 이상 다 가능함
model.add(Dense(units=16, input_shape=(32, ), activation='relu')) 
                        #shpae = (batch_size, input_dim)
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
                                          
                        
#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)
################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras44\\k44_01\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k44_01_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)

start_time=time.time()
hist=model.fit(x_train, y_train, epochs=3000, batch_size=20, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()

#4. predict
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
# accuracy_score = accuracy_score(y_test, y_pre)
# print("ACC_score :", accuracy_score)

#loss : 0.029252858832478523
# acc : 1.0