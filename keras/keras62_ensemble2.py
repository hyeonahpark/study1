import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input

#1. data

x1_datasets = np.array([range(100), range(301,401)]).T
                        #삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
                        #원유, 환율, 금시세                    
x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T


y = np.array(range(3001, 3101)) #한강의 화씨 온도


x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test =train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, train_size=0.9, random_state=5656
)

# print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape) #(90, 2) (90, 3) (90, 4) (90,)
#2-1. model
input1 = Input(shape=(2,))
dense1 = Dense(32, activation='relu', name='bit1')(input1)
dense2 = Dense(64, activation='relu', name='bit2')(dense1)
dense3 = Dense(128, activation='relu', name='bit3')(dense2)
dense4 = Dense(64, activation='relu', name='bit4')(dense3)
output1 = Dense(32, activation='relu', name='bit5')(dense4)
model1 = Model(inputs=input1, outputs = output1)

#2-2. model
input11 = Input(shape=(3,))
dense11 = Dense(32, activation='relu', name='bit11')(input11)
dense21 = Dense(64, activation='relu', name='bit21')(dense11)
dense31 = Dense(128, activation='relu', name='bit31')(dense21)
dense41 = Dense(64, activation='relu', name='bit41')(dense31)
output11 = Dense(32, activation='relu', name='bit51')(dense41)
model11 = Model(inputs=input11, outputs = output11)
model11.summary()

#2-3. model
input111 = Input(shape=(4,))
dense111 = Dense(32, activation='relu', name='bit111')(input111)
dense211 = Dense(64, activation='relu', name='bit211')(dense111)
dense311 = Dense(128, activation='relu', name='bit311')(dense211)
dense411 = Dense(64, activation='relu', name='bit411')(dense311)
output111 = Dense(32, activation='relu', name='bit511')(dense411)
model11 = Model(inputs=input111, outputs = output111)
model11.summary()


#2-3. 합체!!
from keras.layers.merge import Concatenate, concatenate
merge1 = Concatenate()([output1, output11, output111])
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
merge4 = Dense(20, name='mg4')(merge3)
last_output = Dense(1, name = 'last')(merge4)

model = Model(inputs=[input1, input11, input111], outputs = last_output)
model.summary()

model = load_model('_save/keras62/k62_02/k62_020814_1631_0878-0.0000.hdf5')

#3. compile
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)

# ################## mcp 세이브 파일명 만들기 시작 ###################
# import datetime
# date = datetime.datetime.now()
# # print(date) #2024-07-26 16:49:57.565880
# # print(type(date)) #<class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")
# # print(date) #0726_1654
# # print(type(date)) #<class 'str'>


# path = 'C:\\ai5\\_save\\keras62\\k62_02\\'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
# filepath = "".join([path, 'k62_02', date, '_' , filename])
# # #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# # ################## mcp 세이브 파일명 만들기 끝 ###################

# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath=filepath)
# model.fit([x1_train, x2_train, x3_train], y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[mcp])


#4. predict
loss=model.evaluate([x1_test, x2_test, x3_test], y_test)
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)
x1_predict =np.array([range(100,105), range(401,406)]).T
x2_predict =np.array([range(201,206), range(511,516), range(250,255)]).T
x3_predict = np.array([range(100,105), range(401,406), range(177,182), range(133,138)]).T

# print(x1_predict.shape, x2_predict.shape) #(5, 2) (5, 3)
y_predict = model.predict([x1_predict, x2_predict, x3_predict])

print("[3101,3102,3103,3104,3105]예측 : " ,y_predict)
print("loss : ", loss[0])


# [3101,3102,3103,3104,3105]예측 :  [[3101.0005]
#  [3102.0002]
#  [3103.0444]
#  [3104.106 ]
#  [3105.1685]]
# loss :  1.2516974834397843e-07

