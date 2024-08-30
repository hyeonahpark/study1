import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, KFold



#1. data

path = 'C:\\ai5\_data\\kaggle\\bike-sharing-demand\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
# path = 'C://ai5//_data//bike-sharing-demand//' #역슬래시로 작성해도 상관없음
# path = 'C:/ai5/_data/bike-sharing-demand/' 

train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission=pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  #(10886, 11)
# print(test_csv.shape)  #(6493, 8)
# print(sampleSubmission.shape) # (6493, 1)

#casual, registered 는 미등록 사용자와 등록 사용자임. casual+registered 의 수는 count와 동일하므로 두 열을 삭제해도 됨.
# print(train_csv.columns) #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#     # 'humidity', 'windspeed', 'casual', 'registered', 'count'],
#     #   dtype='object')
# print(train_csv.info()) #null 값 확인하기
# print(test_csv.info())

# print(train_csv.describe()) #count, mean, std, min, 1/4분위, 중위값, 3/4분위, max값 나옴. 어떤 주어진 값들을 크기의 순서대로 정렬했을 때 가장 중앙에 위치하는 값

# ################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

################# x와 y 분리 ######################

x=train_csv.drop(['count'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
y=train_csv['count']

# print(train_csv)
# train_csv.boxplot()
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

############################X 데이터 로그변환##########################################
# x['casual'] = np.log1p(x['casual'])
# x['registered'] = np.log1p(x['registered'])
############################X 데이터 로그변환##########################################

##################### y 로그 변환####################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##################### y 로그 변환####################################

#2. modeling
model = RandomForestRegressor(random_state= 52151, max_depth=5, min_samples_split=3)

#3. fit
# model.fit(x_train, y_train)
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# ACC :  [0.99281301 0.99385939 0.99277833 0.992031   0.99363757] 
#  평균 ACC :  0.993