"""
keras49_augument2_mnist.py
keras49_augument3_cifar10.py
keras49_augument4_cifar100.py
keras49_augument5_cat_dog.py
keras49_augument6_men_women.py
keras49_augument7_horse.py
keras49_augument8_rps.py

cat_dog는 image 폴더꺼 수치화하고, 캐글 폴더꺼 수치화해서 합치고 증폭 1만개 추가
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

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
path_image = 'C:\\ai5\\_data\\image\\cat_and_dog\\Train\\'
path_kaggle_cat_dog = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\train\\'
path_test = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\test\\'
path_submission = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'
sample_submission=pd.read_csv(path_submission + "sample_submission.csv", index_col=0)

xy_train1 = test_datagen.flow_from_directory(
    path_image, 
    target_size=(80,80),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 160 images belonging to 2 classes.

xy_train2 = test_datagen.flow_from_directory(
    path_kaggle_cat_dog, 
    target_size=(80,80),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 160 

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(80,80),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  
np_path = 'c:/ai5/_data/_save_npy/augment_cat_dog/'

# np.save(np_path + 'keras49_05_x_train1.npy', arr=xy_train1[0][0])
# np.save(np_path + 'keras49_05_y_train1.npy', arr=xy_train1[0][1])
# np.save(np_path + 'keras49_05_x_train2.npy', arr=xy_train2[0][0])
# np.save(np_path + 'keras49_05_y_train2.npy', arr=xy_train2[0][1])
# np.save(np_path + 'keras49_05_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras49_05_y_test.npy', arr=xy_test[0][1])


x_train1 = np.load(np_path + 'keras49_05_x_train1.npy')
y_train1 = np.load(np_path + 'keras49_05_y_train1.npy')
x_train2 = np.load(np_path + 'keras49_05_x_train2.npy')
y_train2 = np.load(np_path + 'keras49_05_y_train2.npy')
x_test = np.load(np_path + 'keras49_05_x_test.npy')
y_test = np.load(np_path + 'keras49_05_y_test.npy')

x = np.concatenate((x_train1,x_train2))
y = np.concatenate((y_train1,y_train2))

print(x.shape, y.shape) # (44997, 80, 80, 3) (44997,)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

augment_size = 5000 
print(x_train.shape[0])  #40497
randidx=np.random.randint(x_train.shape[0], size = augment_size) # 이미지 1장 # 60000, size = 40000
# print(randidx) # [31998 12497 32753 ... 32228 21276 28073]
# print(np.min(randidx), np.max(randidx)) # randidx의 최솟값과 최댓값 출력 1, 59991 

print(x_train[0].shape) # (80, 80, 3)

x_augmented = x_train[randidx].copy() #.copy하면 메모리를 따로 할당하므로 원래 있던 x_train 값에 영향을 미치지 않음.
y_augmented = y_train[randidx].copy() 
print(x_augmented.shape, y_augmented.shape) # (5000, 80, 80, 3) (5000,)
 
x_augmented = x_augmented.reshape(
    x_augmented.shape[0], #
    x_augmented.shape[1], #
    x_augmented.shape[2], 3 #
)

print(x_augmented.shape) # (5000, 80, 80, 3)

 
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c:/ai5/_data/_save_img/05_cat_dog/'
).next()[0]

"""
print(x_augmented.shape) # (5000, 80, 80, 3)

x_train = x_train.reshape(40497,80,80,3)
x_test = x_test.reshape(4500,80,80,3)

print(x_train.shape, x_test.shape) #(40497, 80, 80, 3) (4500, 80, 80, 3)
 
x_train = np.concatenate((x_train, x_augmented))
print(x_train.shape) #(45497, 80, 80, 3)

y_train = np.concatenate((y_train, y_augmented))
print(y_train.shape) #(45497,)


xy_test=xy_test[0][0]

# #2. modeling
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


path = 'C:\\ai5\\_save\\keras49\\k49_05_cat_dog\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k49_05_', date, '_' , filename])
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
# model.fit_generator(x_train, y_train,
#                     epochs=1000,
#                     verbose=1,
#                     callbacks=[es, mcp],
#                     validation_steps=50)
end_time=time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = np.round(model.predict(x_test, batch_size=16))
print("걸린 시간 :", round(end_time-start_time,2),'초')

# y_pre = np.round(y_pre)

### csv 파일 만들기 ###
y_submit = model.predict(xy_test, batch_size=16)
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
# print(y_submit)

# y_submit = np.round(y_submit)
# print(y_submit)

# print(y_submit)
sample_submission['label'] = y_submit
sample_submission.to_csv(path_submission + "sampleSubmission_0806_02.csv")

"""