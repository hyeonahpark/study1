#https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv

#scaling 작업 -> 몰려있는 값 표준편차로 구하기

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd

#1. data

path = 'C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)  # (165034, 13)
print(test_csv.shape)  # (110023,12)
print(sample_submission.shape) #(110023,1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

x=train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
test_csv=test_csv.drop(['CustomerId', 'Surname'], axis=1)

y=train_csv['Exited']


from sklearn.preprocessing import MinMaxScaler, RobustScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])
test_csv[:] = scalar.fit_transform(test_csv[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

# gbrt = GradientBoostingClassifier(random_state=0)
# gbrt.fit(x_train, y_train)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler=RobustScaler()
# scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. modeling
model=Sequential()
model.add(Dense(32, input_dim=10, activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=20,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time()
model.fit(x_train, y_train, epochs=2000, batch_size=32, verbose=1, validation_split=0.1, callbacks=[es])
end_time=time.time()

#4. predict
loss=model.evaluate(x_test, y_test, verbose = 1)


y_predict = model.predict(x_test)
y_predict = np.round(y_predict)  # 사이킷런의 acc 평가지표는 정수만 받음. 분류 데이터는 분류 값만 넣으라는 에러 발생, 따라서 반올림함.
# print(y_predict)



y_submit = np.round(model.predict(test_csv)) # type: ignore

print(y_submit)
print(y_submit.shape) #(116, 1)

############  submission.csv 만들기 // count 컬럼에 값 넣어주기

sample_submission['Exited'] = y_submit
# print(sample_submission) 
# print(sample_submission.shape) # (116, 2)from sklearn.metrics import r2_score

sample_submission.to_csv(path + "submission_0725_3.csv")


from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(y_test, y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("acc_score : ", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")



# loss :  0.5007482767105103,ACC :  0.793


# binary
#loss :  0.3231148421764374, ACC :  0.863


#minmaxscaler
#loss :  0.3233070373535156. ACC :  0.863

#standardscaler
#loss :  0.32407858967781067, ACC :  0.863

#maxabs
#loss :  0.3221178352832794 ACC :  0.863

#robust
#loss :  0.32438650727272034, ACC :  0.862