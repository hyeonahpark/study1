import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#1. data

path = "C:\\ai5\\_data\\dacon\\따릉이\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
print(submission_csv) # [715 rows x 1 columns]


######################결측치 처리 1. 삭제 #############################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum())
train_csv=train_csv.dropna() #결측치 포함 행 제거
print(train_csv.isna().sum())
print(train_csv)

print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean()) #fillna 함수 : 결측치를 채운다
print(test_csv.info())

x = train_csv.drop(['count'], axis=1) #train_csv에서 count 열 삭제 후 x에 넣기
y = train_csv['count'] #train_csv에서 count 열만 y에 넣기

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)

############################X 데이터 로그변환##########################################
x['hour_bef_visibility'] = np.log1p(x['hour_bef_visibility'])
############################X 데이터 로그변환##########################################

##################### y 로그 변환####################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##################### y 로그 변환####################################
 

#[실습]
#R2 0.62 이상

#2. model 랜덤 포레스트 모델 학습
model = RandomForestRegressor(random_state= 52151, max_depth=5, min_samples_split=3)

#3. fit
model.fit(x_train, y_train)

#4. predict
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print("score : ", score) 

#변환 x  score :  0.7546419466990388
#x만 변환 score :  0.7546419466990388
#y만 변환 score :  0.7292518333401288
#둘다변환 score :  0.7292518333401288