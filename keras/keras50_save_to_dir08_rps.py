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

train_datagen =  ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=0.2,              # 축소 또는 확대
    shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

path = './_data/image/rps/'

xy_train = train_datagen.flow_from_directory(
    path, 
    target_size=(300,300),
    batch_size=1100,
    class_mode='sparse', #원핫도 되서 나옴, sparse는 원핫 이전 상태로 나옴
    color_mode='rgb', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 160 images belonging to 2 classes.

start_time=time.time()
np_path = 'c:/ai5/_data/_save_npy/rps/'
x_train=np.load(np_path + 'keras49_08_x_train.npy')
y_train=np.load(np_path + 'keras49_08_y_train.npy')


end_time=time.time()
# print("데이터 불러오는 시간 :", round(end_time-start_time,2),'초') #146.38초
# print(x_train)
# print(x_train.shape) #(1100, 300, 300, 3)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=5656)


augment_size = 2000 
 
# print(x_train.shape[0])  #990
randidx=np.random.randint(x_train.shape[0], size = augment_size) # 이미지 1장 # 60000, size = 40000
# print(randidx) # [31998 12497 32753 ... 32228 21276 28073]
# print(np.min(randidx), np.max(randidx)) # randidx의 최솟값과 최댓값 출력 1, 59991 

# print(x_train[0].shape) # (300, 300, 3)

x_augmented = x_train[randidx].copy() #.copy하면 메모리를 따로 할당하므로 원래 있던 x_train 값에 영향을 미치지 않음.
y_augmented = y_train[randidx].copy() 
# print(x_augmented.shape, y_augmented.shape) # (5000, 300, 300, 3) (5000, )
 
x_augmented = x_augmented.reshape(
    x_augmented.shape[0], #10000
    x_augmented.shape[1], #300
    x_augmented.shape[2], 3 #300
)

# print(x_augmented.shape) #(5000, 300, 300, 3)

 
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/ai5/_data/_save_img/08_rps/'
).next()[0]

"""
# print(x_augmented.shape) # ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape', (40000, 28, 28))
# print(x_augmented.shape) # (5000, 300, 300, 3)

x_train = x_train.reshape(990,300,300,3)
x_test = x_test.reshape(110,300,300,3)

# print(x_train.shape, x_test.shape) #(990, 300, 300, 3) (110, 300, 300, 3)
 
x_train = np.concatenate((x_train, x_augmented))
# print(x_train.shape) #(5990, 300, 300, 3))

y_train = np.concatenate((y_train, y_augmented))
# print(y_train.shape) #(5990, )

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #sparse=True가 기본값
y_train= ohe.fit_transform(y_train.reshape(-1,1))
y_test= ohe.fit_transform(y_test.reshape(-1,1))

# print(x_train.shape, y_train.shape) # (5990, 300, 300, 3) (5990, 3)
# print(x_test.shape, y_test.shape) # (110, 300, 300, 3) (110, 3)


# #2. modeling
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3), padding='same')) 
                        #shape = (batch_size, rows, columns, channels) #batch_size : 훈련시킬 데이터의 갯수
                        #shape = (batch_size, heights, widths, channels) #다음에 넘어갈 때는 height, widhts, filter 로 받아들임
model.add(MaxPool2D())
model.add(Conv2D(filters=32, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Flatten()) 
model.add(Dense(units=32, activation='relu')) 
model.add(Dense(units=16, input_shape=(32, ), activation='relu')) 
                        #shpae = (batch_size, input_dim)
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
                                          
                        
#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
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


path = 'C:\\ai5\\_save\\keras49\\k49_08\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k49_08_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
hist=model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.3, callbacks=[es, mcp])
end_time=time.time()


#4. predict
from sklearn.metrics import r2_score, accuracy_score


loss = model.evaluate(x_test, y_test, verbose=1, batch_size=5)
print('loss :', loss[0])
print('acc :', round(loss[1], 5))

# y_pre = np.round(model.predict(x_test))
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end_time-start_time,2),'초')
# r2=r2_score(y_test, y_pre)
# accuracy_score = accuracy_score(y_test, y_pre)
# print("ACC_score :", accuracy_score)



#loss : 0.04563869908452034
# acc : 0.98182


# loss : 0.00012304952542763203
# acc : 1.0

#augment
# loss : 0.05186311528086662
# acc : 0.97273

# loss : 0.0007380721508525312
# acc : 1.0
"""