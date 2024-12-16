#https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv


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

################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

# print(test_csv.info())
# print(train_csv.describe())

################# x와 y 분리 ######################
# x=train_csv.drop(['CustomerId', 'Surname'], axis=1)
# # print(x) #(165034, 11)
# # print(x.shape) #(165034, 11)

# y=train_csv['Exited']
# print(y.shape) # (165034, 1)
# print(test_csv.info())
# print(train_csv.info())

from tensorflow.keras.preprocessing.text import Tokenizer

train_geography = train_csv['Geography']
train_gender = train_csv['Gender']
test_geography = test_csv['Geography']
test_gender = test_csv['Gender']

# print(test_csv['Gender'])
# print(test_gender)

tokenizer_geography_train= Tokenizer()
tokenizer_gender_train= Tokenizer()
tokenizer_geography_test= Tokenizer()
tokenizer_gender_test= Tokenizer()
tokenizer_geography_train.fit_on_texts(train_geography)
tokenizer_gender_train.fit_on_texts(train_gender)
tokenizer_geography_test.fit_on_texts(test_geography)
tokenizer_gender_test.fit_on_texts(test_gender)

#train_csv의 tokenizer
print(tokenizer_geography_train.word_index)  # {'france': 1, 'spain': 2, 'germany': 3}
print(tokenizer_geography_train.word_counts) # OrderedDict([('france', 94215), ('spain', 36213), ('germany', 34606)])
print(tokenizer_geography_train.texts_to_sequences(train_geography))

print(tokenizer_gender_train.word_index)  # {'male': 1, 'female': 2}
print(tokenizer_gender_train.word_counts) # OrderedDict([('male', 93150), ('female', 71884)])
print(tokenizer_gender_train.texts_to_sequences(train_gender))

#test_csv의 tokenizer
print(tokenizer_geography_test.word_index)  # {'france': 1, 'spain': 2, 'germany': 3}
print(tokenizer_geography_test.word_counts) # OrderedDict([('france', 157386), ('spain', 60126), ('germany', 57545)])
print(tokenizer_geography_test.texts_to_sequences(test_geography))

print(tokenizer_gender_test.word_index)  # {'male': 1, 'female': 2}
print(tokenizer_gender_test.word_counts) # OrderedDict([('male', 155092), ('female', 119965)])
print(tokenizer_gender_test.texts_to_sequences(test_gender))


#문자열 -> 수치화 한 후 데이터 값 변환
train_csv['Geography']=tokenizer_gender_train.texts_to_sequences(train_geography)
train_csv['Gender']=tokenizer_gender_train.texts_to_sequences(train_gender)
test_csv['Geography']=tokenizer_gender_test.texts_to_sequences(test_geography)
test_csv['Gender']=tokenizer_gender_test.texts_to_sequences(test_gender)

x=train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y=train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=512)

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
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', #모르면 auto
    patience=5,
    restore_best_weights=True, #작성 안하면 마지막 지점 반환/ True인 경우 가장 좋은 weight 사용
    )


model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
start_time=time.time()
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1, validation_split=0.1, callbacks=[es])
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

sample_submission['Exited'] = y_submit
# print(sample_submission) 
# print(sample_submission.shape) # (116, 2)from sklearn.metrics import r2_score

sample_submission.to_csv(path + "submission_0722_1.csv")


from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(y_test, y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

print("loss : ", loss[0])
print("ACC : ", round(loss[1], 3))
# print("acc_score : ", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")
