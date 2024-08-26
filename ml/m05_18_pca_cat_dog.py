# 배치를 160으로 잡고
# x,y를 추출해서 모델을 맹그러라
# acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

#1. data (시간체크)
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

path_train = './_data/image/cat_and_dog/Train/'
path_test = './_data/image/cat_and_dog/Test/'

# xy_train = train_datagen.flow_from_directory(
#     path_train, 
#     target_size=(100,100),
#     batch_size=20000,
#     class_mode='binary',
#     color_mode='rgb', 
#     shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
# )  # Found 160 images belonging to 2 classes.

# xy_test = test_datagen.flow_from_directory(
#     path_test, 
#     target_size=(100,100),
#     batch_size=20000,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=False, # 어지간하면 셔플할 필요 없음.
# )  # Found 120 images belonging to 2 classes.

np_path = 'c:/ai5/_data/_save_npy/augment_cat_dog/'
x_train1 = np.load(np_path + 'keras49_05_x_train1.npy')
y_train1 = np.load(np_path + 'keras49_05_y_train1.npy')
x_test = np.load(np_path + 'keras49_05_x_test.npy')

print(x_train1.shape, y_train1.shape) #(19997, 80, 80, 3) (19997,)

x = x_train1.reshape(19997,80*80*3)
y = y_train1

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=5656)

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
evr_cumsum = np.cumsum(evr)
n = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

for i in range(0, len(n), 1) :
    pca = PCA(n_components = n[i])
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.transform(x_test)  

    # #2. modeling
    model = Sequential()
    model.add(Dense(64, input_shape=(n[i], ), activation='relu')) 
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) 
                                            
                            
    #3. compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
    start_time=time.time()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True)

    ################## mcp 세이브 파일명 만들기 시작 ###################
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")


    path_save = 'C:\\ai5\\_save\\m05\\m05_18\\'
    filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
    filepath = "".join([path_save, 'm05_18_', str(i+1), '_', date, '_' , filename])
    #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
    ################## mcp 세이브 파일명 만들기 끝 ###################

    mcp=ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose = 0,
        save_best_only=True,
        filepath=filepath)

    hist=model.fit(x_train2, y_train, epochs=1000, batch_size=1024, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end_time=time.time()


    #4. predict
    from sklearn.metrics import r2_score, accuracy_score


    loss = model.evaluate(x_test2, y_test, verbose=1)
    y_pre = np.round(model.predict(x_test2))
    print("##############################################")
    print("결과.PCA :", n[i])
    print("loss : ", loss[0])
    print("ACC : ", round(loss[1], 6))
    print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수
    


# loss : 0.28785383701324463
# acc : 0.884

# loss : 0.28397536277770996
# acc : 0.8765

# loss : 0.29588955640792847
# acc : 0.8815

#cnn1d
# loss : 0.6740598678588867
# acc : 0.588
# 걸린 시간 : 191.22 초

##############################################
# 결과.PCA : 1524
# loss :  0.6709826588630676
# ACC :  0.5915
# 걸린 시간 :  1.6 초       
# 63/63 [==============================] - 0s 2ms/step - loss: 0.6568 - accuracy: 0.6100 - acc: 0.6100 - mse: 0.2323
# ##############################################
# 결과.PCA : 4601
# loss :  0.6567885875701904
# ACC :  0.61
# 걸린 시간 :  1.94 초
# 63/63 [==============================] - 0s 3ms/step - loss: 0.6574 - accuracy: 0.6150 - acc: 0.6150 - mse: 0.2323
# ##############################################
# 결과.PCA : 8115
# loss :  0.6573594212532043
# ACC :  0.615
# 걸린 시간 :  2.04 초
# 63/63 [==============================] - 0s 3ms/step - loss: 0.6507 - accuracy: 0.6250 - acc: 0.6250 - mse: 0.2295
# ##############################################
# 결과.PCA : 13485
# loss :  0.6507197618484497
# ACC :  0.625
# 걸린 시간 :  3.02 초