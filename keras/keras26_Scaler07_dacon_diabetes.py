#https://dacon.io/competitions/official/236068

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. data

path = 'C:\\ai5\\_data\\dacon\\diabetes\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)  # (652,9)
print(test_csv.shape)  # (116,8)
print(sample_submission.shape) #(116,1)

################## 결측치 확인 #####################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

###########################################

print(test_csv.info())

print(train_csv.describe())

################# x와 y 분리 ######################
x=train_csv.drop(['Outcome'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
print(x)
print(x.shape) #(652, 8)
y=train_csv['Outcome']
print(y.shape) # (652., )

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

print("고유한 요소:", unique) #고유한 요소: [0 1]
print("각 요소의 개수:", counts) #각 요소의 개수: [424 228]


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler=RobustScaler()
scaler.fit(x)
x = scaler.transform(x) 



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)


#2. modeling
model=Sequential()
model.add(Dense(16, input_dim=8, activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=70,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )

import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>

path = 'C:\\ai5\\_save\\keras30_mcp\\k30_7\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k30_7_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ###################

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, validation_split=0.1, callbacks=[es, mcp])
end_time=time.time()

#4. predict
loss=model.evaluate(x_test, y_test, verbose = 1)


y_predict = model.predict(x_test)
y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_predict)



y_submit = np.round(model.predict(test_csv))
# print(y_submit)
# print(y_submit.shape) #(116, 1)

#############  submission.csv 만들기 // count 컬럼에 값 넣어주기

sample_submission['Outcome'] = y_submit
# print(sample_submission) 
# print(sample_submission.shape) # (116, 2)from sklearn.metrics import r2_score

sample_submission.to_csv(path + "submission_0813.csv")


from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(y_test, y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("acc_score : ", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")


# mse 사용
#loss :  0.16671177744865417, ACC :  0.788, 걸린 시간 :  38.07 초
#loss :  0.16646991670131683, ACC :  0.758. 걸린 시간 :  33.83 초

# binary_cross entropy 사용
#loss :  0.5410665273666382, ACC :  0.758, 걸린 시간 :  37.37 초
#loss :  0.5549544095993042, ACC :  0.773, 걸린 시간 :  27.63 초
#loss :  0.6763560771942139, ACC :  0.773, 걸린 시간 :  63.31 초
#loss :  0.6377341151237488, ACC :  0.773
#loss :  0.3791206181049347, ACC :  0.818, 걸린 시간 :  48.73 초
#loss :  0.44093620777130127, ACC :  0.788. 걸린 시간 :  23.02 초
#loss :  0.4059293866157532, ACC :  0.848, 걸린 시간 :  28.05 초


#minmaxscaler
#loss :  0.4352138042449951, ACC :  0.833

#standardscaler
#loss :  0.4277765154838562, ACC :  0.833

#maxabs
#loss :  0.44795262813568115 ACC :  0.788

#robust
# loss :  0.37251511216163635, ACC :  0.864