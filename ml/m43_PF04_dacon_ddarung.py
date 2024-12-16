# Pseudo Labeling 기법 : 모델 돌려서 나온 결과로 결측치를 찾아

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier
import pandas as pd 
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import PolynomialFeatures

#1. data

path = "C:\\ai5\\_data\\dacon\\따릉이\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
test_csv = pd.read_csv(path + "test.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함
submission_csv = pd.read_csv(path + "submission.csv", index_col=0) #. 하나는 root 라는 뜻, 그 하단은 /로 표현, index_col=0을 해주면 0번째인 id가 인덱스라는 것을 표현함


train_csv=train_csv.dropna() #결측치 포함 행 제거
test_csv = test_csv.fillna(test_csv.mean()) #fillna 함수 : 결측치를 채운다

x = train_csv.drop(['count'], axis=1) #train_csv에서 count 열 삭제 후 x에 넣기
y = train_csv['count'] #train_csv에서 count 열만 y에 넣기

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)

############################X 데이터 로그변환##########################################
x['hour_bef_visibility'] = np.log1p(x['hour_bef_visibility'])
############################X 데이터 로그변환##########################################

pf = PolynomialFeatures(degree = 2, include_bias=False)
x = pf.fit_transform(x)

##################### y 로그 변환####################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##################### y 로그 변환####################################
 

random_state = 1199
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)


#2. model
model = RandomForestRegressor(random_state= 52151, max_depth=5, min_samples_split=3)


#3. 훈련
model.fit(x_train, y_train)

#4. predict
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print("score : ", score) 

# model.score :  0.7878787878787878
# 스태킹 acc :  0.7878787878787878

# -------------PF -------------
# model.score :  0.7727272727272727
# 스태킹 acc :  0.7727272727272727