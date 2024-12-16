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


y1 = np.array(range(3001, 3101)) #한강의 화씨 온도
y2 = np.array(range(13001,13101)) #비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test =train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.9, random_state=5656
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
merge2 = Dense(20, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
middle_output = Dense(20, name='mg4')(merge3)

#2-4. 분기1

# dense441= Dense(32,  activation='relu')(middle_output)
# dense442 = Dense(64, activation='relu')(dense441)
# dense443 = Dense(128, activation='relu')(dense442)
# dense444 = Dense(64, activation='relu')(dense443)
# dense445 = Dense(32, activation='relu')(dense444)
# last_output1 = Dense(1, activation='relu')(dense445)

# #2-5. 분기2
# dense551= Dense(32,  activation='relu')(middle_output)
# dense552 = Dense(64, activation='relu')(dense551)
# dense553 = Dense(128, activation='relu')(dense552)
# dense554 = Dense(64, activation='relu')(dense553)
# dense555 = Dense(32, activation='relu')(dense554)
# last_output2 = Dense(1, activation='relu')(middle_output)

last_output1 = Dense(1, name = 'last')(middle_output)
last_output2 = Dense(1, name = 'last2')(middle_output)


model = load_model('C:\\ai5\\_save\\keras62\\k62_03\\k62_0814_1601_0578-0.0000.hdf5')

# model = Model(inputs=[input1, input11, input111], outputs = [last_output1 , last_output2])
model.summary()

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


# path = 'C:\\ai5\\_save\\keras62\\k62_03\\'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
# filepath = "".join([path, 'k62_', date, '_' , filename])
# # #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# # ################## mcp 세이브 파일명 만들기 끝 ###################

# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath=filepath)

# model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=1000, batch_size=1, validation_split=0.2, callbacks=[mcp])


#4. predict
loss=model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)
x1_predict =np.array([range(100,105), range(401,406)]).T
x2_predict =np.array([range(201,206), range(511,516), range(250,255)]).T
x3_predict = np.array([range(100,105), range(401,406), range(177,182), range(133,138)]).T

# print(x1_predict.shape, x2_predict.shape) #(5, 2) (5, 3)
y_predict = model.predict([x1_predict, x2_predict, x3_predict])

print("[3101,3102,3103,3104,3105], [13101,13102,13103,13104,13105] 예측 : ", y_predict)
print("loss : ", loss[0])


# [3101,3102,3103,3104,3105], [13101,13102,13103,13104,13105] 예측 :  [array([[3101.0007],
#        [3102.0078],
#        [3103.0864],
#        [3104.0815],
#        [3105.6475]], dtype=float32), array([[13100.999],
#        [13102.031],
#        [13103.411],
#        [13104.416],
#        [13108.217]], dtype=float32)]
# loss :  2.3007394247542834e-06