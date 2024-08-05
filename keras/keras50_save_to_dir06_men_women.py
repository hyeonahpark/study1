"""
keras49_augument2_mnist.py
keras49_augument3_cifar10.py
keras49_augument4_cifar100.py
keras49_augument5_cat_dog.py
keras49_augument6_men_women.py
keras49_augument7_horse.py
keras49_augument8_rps.py
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

path_train = 'C:\\ai5\\_data\\kaggle\\biggest_gender\\faces\\'
path_test = 'C:\\ai5\\_data\\image\\me\\'

train_datagen =  ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=1,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=0.2,              # 축소 또는 확대
    shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

xy_train = test_datagen.flow_from_directory(
    path_train, 
    target_size=(80,80),
    batch_size=30000,
    class_mode='binary',
    color_mode='rgb', 
    shuffle=False, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(80,80),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False, # 어지간하면 셔플할 필요 없음.
)  #
np_path = 'C:\\ai5\\_data\\_save_npy\\biggest_gender\\'
# print(xy_train[0][0].shape) # (27167, 80, 80, 3)
# print(xy_train[0][1].shape) # (27167, )

# print(xy_train[0][0][:17678].shape) # (17678, 80, 80, 3)
# print(xy_train[0][0][17678:].shape) # (9489, 80, 80, 3)
# np.save(np_path + 'keras49_06_x_train_man.npy', arr=xy_train[0][0][:17678])
# np.save(np_path + 'keras49_06_y_train_man.npy', arr=xy_train[0][1][:17678])
# np.save(np_path + 'keras49_06_x_train_woman.npy', arr=xy_train[0][0][17678:])
# np.save(np_path + 'keras49_06_y_train_woman.npy', arr=xy_train[0][1][17678:])

x_train_man=np.load(np_path + 'keras49_06_x_train_man.npy')
y_train_man=np.load(np_path + 'keras49_06_y_train_man.npy')
x_train_woman=np.load(np_path + 'keras49_06_x_train_woman.npy')
y_train_woman=np.load(np_path + 'keras49_06_y_train_woman.npy')
x_test2=np.load(np_path + 'keras45_07_x_test2.npy')
y_test2=np.load(np_path + 'keras45_07_y_test2.npy')

augment_size = 9000 
 
print(x_train_woman.shape[0])  #9489
randidx=np.random.randint(x_train_woman.shape[0], size = augment_size) # 이미지 1장 # 60000, size = 40000
# print(randidx) # [31998 12497 32753 ... 32228 21276 28073]
# print(np.min(randidx), np.max(randidx)) # randidx의 최솟값과 최댓값 출력 1, 59991 

print(x_train_woman[0].shape) # (80, 80, 3)

x_augmented = x_train_woman[randidx].copy() #.copy하면 메모리를 따로 할당하므로 원래 있던 x_train 값에 영향을 미치지 않음.
y_augmented = y_train_woman[randidx].copy() 
print(x_augmented.shape, y_augmented.shape) # (9000, 80, 80, 3) (9000,)
 
x_augmented = x_augmented.reshape(
    x_augmented.shape[0], #40000
    x_augmented.shape[1], #28
    x_augmented.shape[2], 3 #28
)

print(x_augmented.shape) # (9000, 80, 80, 3)
 
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/ai5/_data/_save_img/06_men_women/'
).next()[0]

"""
x_train_woman = np.concatenate((x_train_woman, x_augmented))
print(x_train_woman.shape) # (18489, 80, 80, 3)

y_train_woman = np.concatenate((y_train_woman, y_augmented))
print(y_train_woman.shape) # (18489,)

x = np.concatenate((x_train_man, x_train_woman))
y = np.concatenate((y_train_man, y_train_woman))

print(x.shape, y.shape) #(36167, 80, 80, 3) (36167,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

#2. modeling
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(80, 80, 3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))
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


path = 'C:\\ai5\\_save\\keras49\\k49_06\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k49_06_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ################### 

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es, mcp])


# model=load_model('./_save/keras45/k45_07/k45_07_0805_1439_0112-0.1637.hdf5')

end_time=time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

# y_pre = np.round(model.predict(x_test, batch_size=16))
print("걸린 시간 :", round(end_time-start_time,2),'초')

y_submit = model.predict(x_test2, batch_size=16)
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
print(y_submit)
"""