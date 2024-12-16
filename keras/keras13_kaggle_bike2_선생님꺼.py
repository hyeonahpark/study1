# 기존 kaggle 데이터에서 
# 1. train_cav의 y를 casual과 registered로 잡는다.
#    그래서 훈련을 해서 test_cav의 casual과 registered를 predict 한다. 

# 2. test_csv에 casual과 registered 컬럼을 합친다 (파일을 만듦)

# 3. train_csv에 y를 count로 잡는다. 

# 4. 전체 훈련

# 5. test_csv 예측해서 submission에 붙인다. 

# #
# 1. 우리가 원하는 특정 컬럼을 생성하는 과정
# 2. 생성한 컬럼을 가지고 훈련하는 과정

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. data

path = 'C:\\ai5\_data\\bike-sharing-demand\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
# path = 'C://ai5//_data//bike-sharing-demand//' #역슬래시로 작성해도 상관없음
# path = 'C:/ai5/_data/bike-sharing-demand/' 

train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission=pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  #(10886, 11)
print(test_csv.shape)  #(6493, 8)
print(sampleSubmission.shape) # (6493, 1)

#casual, registered 는 미등록 사용자와 등록 사용자임. casual+registered 의 수는 count와 동일하므로 두 열을 삭제해도 됨.
print(train_csv.columns) #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    # 'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(train_csv.info()) #null 값 확인하기
print(test_csv.info())

print(train_csv.describe()) #count, mean, std, min, 1/4분위, 중위값, 3/4분위, max값 나옴. 어떤 주어진 값들을 크기의 순서대로 정렬했을 때 가장 중앙에 위치하는 값

################## 결측치 확인 #####################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

################# x와 y 분리 ######################

x=train_csv.drop(['casual', 'registered', 'count'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
print(x)
print(x.shape) #(10886, 8)

y=train_csv[['casual', 'registered']]
print(y)
print(y.shape) # (10886, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=123)

#2. modeling
model=Sequential()
model.add(Dense(64, input_dim=8, activation='relu')) #activation function 활성화 함수, 한정함수 : 다음레이어에 오는 값의 범위를 한정한다. y=relu(wx+b) , relu 함수는 0보다 낮은 값이 나오면 0으로 나옴.
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. predict
loss=model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("R2의 점수 : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)

print(test_csv.shape)
print(y_submit.shape) 


print("test_csv타입 : ", type(test_csv)) # test_csv 타입 : <class 'pandas.core.frame.DataFrame'>
print("y_submit타입 : ", type(y_submit)) # <class 'numpy.ndarray'>

test2_csv = test_csv #원래는 .copy 사용 해야함
print(test2_csv.shape) # (6493, 8)

test2_csv[['casual', 'registered']] = y_submit
print(y_submit)
print(test2_csv) #[6493 rows * 10 columns]

test2_csv.to_csv(path + "test2.csv")