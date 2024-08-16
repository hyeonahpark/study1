import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input

#1. data

x_datasets = np.array([range(100), range(301,401)]).T
                        #삼성 종가, 하이닉스 종가

y1 = np.array(range(3001, 3101)) #한강의 화씨 온도
y2 = np.array(range(13001,13101)) #비트코인 가격


x_train, x_test, y1_train, y1_test, y2_train, y2_test =train_test_split(
    x_datasets, y1, y2, train_size=0.9, random_state=5656
)

print(x_train.shape, y1_train.shape,  y2_train.shape) #(90, 2) (90,) (90,)

#2-1. model
input1 = Input(shape=(2,))
dense1 = Dense(32, activation='relu', name='bit1')(input1)
dense2 = Dense(64, activation='relu', name='bit2')(dense1)
dense3 = Dense(128, activation='relu', name='bit3')(dense2)
dense4 = Dense(64, activation='relu', name='bit4')(dense3)
output1 = Dense(32, activation='relu', name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs = output1)


#2-2. output
last_output1 = Dense(1, name = 'last')(output1)
last_output2 = Dense(1, name = 'last2')(output1)

model = Model(inputs=input1, outputs = [last_output1 , last_output2])

model = load_model('_save/keras62/k62_04/k62_040814_1620_0812-0.0001.hdf5')

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


# path = 'C:\\ai5\\_save\\keras62\\k62_04\\'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
# filepath = "".join([path, 'k62_04', date, '_' , filename])
# # #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# # ################## mcp 세이브 파일명 만들기 끝 ###################

# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath=filepath)

# model.fit(x_train, [y1_train, y2_train], epochs=1000, batch_size=1, validation_split=0.2, callbacks=[mcp])


#4. predict
loss=model.evaluate(x_test, [y1_test, y2_test])
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)
x_predict =np.array([range(100,105), range(401,406)]).T

# print(x1_predict.shape, x2_predict.shape) #(5, 2) (5, 3)
y_predict = model.predict(x_predict)

print("[3101,3102,3103,3104,3105], [13101,13102,13103,13104,13105] 예측 : ", y_predict)
print("loss : ", loss[0])

# [3101,3102,3103,3104,3105], [13101,13102,13103,13104,13105] 예측 :  [array([[3100.6748],
#        [3101.4338],
#        [3102.1929],
#        [3102.9514],
#        [3103.7102]], dtype=float32), array([[13101.109],
#        [13102.193],
#        [13103.277],
#        [13104.359],
#        [13105.444]], dtype=float32)]
# loss :  7.551311864517629e-05