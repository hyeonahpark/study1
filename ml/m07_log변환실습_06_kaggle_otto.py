import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path +"sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  # (61878, 94)
# print(test_csv.shape)  # (144368, 93)
# print(sampleSubmission_csv.shape) # (144368, 9)

################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

################# x와 y 분리 ######################
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])
x=train_csv.drop(['target'], axis=1)
y=train_csv['target']

# print(train_csv)
# train_csv.boxplot()
# plt.show()

unique,counts=np.unique(y, return_counts=True)
y=pd.get_dummies(y)




############################X 데이터 로그변환##########################################
# x['feat_24'] = np.log1p(x['feat_24'])
# x['feat_73'] = np.log1p(x['feat_73'])
# x['feat_74'] = np.log1p(x['feat_74'])
############################X 데이터 로그변환##########################################

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

############################X 데이터 로그변환##########################################
# x['feat_24'] = np.log1p(x['feat_24'])
# x['feat_73'] = np.log1p(x['feat_73'])
# x['feat_74'] = np.log1p(x['feat_74'])
############################X 데이터 로그변환##########################################


##################### y 로그 변환####################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##################### y 로그 변환####################################

#2. model 랜덤 포레스트 모델 학습
model = RandomForestRegressor(random_state= 1186, max_depth=5, min_samples_split=3)

#3. fit
model.fit(x_train, y_train)

#4. predict
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print("score : ", score) 

#변환 x   score :  0.2665403294545532
#x만 변환 score :  0.26654036675128734
#y만 변환 score :  0.26664593216209576
#둘다변환 score :  0.2666459697633407