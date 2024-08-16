import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input

#1. data

x1_datasets = np.array([range(100), range(301,401)]).T
                        #삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
                        #원유, 환율, 금시세
                        
y = np.array(range(3001, 3101)) #한강의 화씨 온도




# print(x1_datasets.shape, x2_datasets.shape, y.shape) #(100, 2) (100, 3) (100,)


# x_train, x_test, y_train, y_test = train_test_split(x1_datasets, y, train_size=0.9, random_state=5656)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(x2_datasets, y, train_size=0.9, random_state=5656)

x1_train, x1_test, x2_train, x2_test, y_train, y_test =train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.9, random_state=5656
)

print(x1_train.shape, x2_train.shape, y_train.shape) #(70, 2) (70, 3) (70,)


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

#2-3. 합체!!
from keras.layers.merge import Concatenate, concatenate
# merge1 = Concatenate()([dense4, dense21])
# model1 = Dense(10)(merge1)
# model11 = Dense(5)(model1)
# output = Dense(1)(model11)
# model = Model(inputs = [input1, input11], outputs = output)
# model.summary()


#쌤이 해준고
merge1 = Concatenate()([output1, output11])
# merge1 = concatenate([output1, output11], name = 'mg1')
# merge2 = Dense(7, name='mg2')(merge1)
# merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge1)

model = Model(inputs=[input1, input11], outputs = last_output)
model.summary()


model = load_model('_save/keras62/k62_01/k62_010814_1630_0781-0.0000.hdf5')

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


# path = 'C:\\ai5\\_save\\keras62\\k62_01\\'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
# filepath = "".join([path, 'k62_01', date, '_' , filename])
# # #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# # ################## mcp 세이브 파일명 만들기 끝 ###################

# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath=filepath)

# model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[mcp])


#4. predict

loss=model.evaluate([x1_test, x2_test], y_test)
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)
x1_predict =np.array([range(100,105), range(401,406)]).T
x2_predict =np.array([range(201,206), range(511,516), range(250,255)]).T

# print(x1_predict.shape, x2_predict.shape) #(5, 2) (5, 3)
y_predict = model.predict([x1_predict, x2_predict])

print("[3101,3102,3103,3104,3105]예측 : " ,y_predict)


print("loss : ", loss[0])
# print("ACC : ", round(loss[1], 3))
# print("걸린 시간 : ", round(end_time - start_time, 2), "초") # round 함수 : 반올림, 뒤에 숫자는 소수 자리 수


# [3101,3102,3103,3104,3105]예측 :  [[3101.0125]
#  [3102.0288]
#  [3103.044 ]
#  [3104.0596]
#  [3105.075 ]]
# loss :  2.3841858265427618e-08