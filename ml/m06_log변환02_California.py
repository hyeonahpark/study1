from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#1. data
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# print(df)

df['target'] = datasets.target

# df.boxplot()
# plt.show()

# print(df.info())
# print(df.describe())

# df['Population'].boxplot() #안됨
# df['Population'].plot.boxplot() #됨
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

############################Population 로그변환##########################################
# x['Population'] = np.log1p(x['Population']) #지수변환  np.expm1
############################Population 로그변환##########################################

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1186)

##################### y 로그 변환####################################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
##################### y 로그 변환####################################

#2. model 랜덤 포레스트 모델 학습
# model = RandomForestRegressor(random_state= 1186, max_depth=5, min_samples_split=3)


model = LinearRegression()


#3. fit
model.fit(x_train, y_train)

#4. predict
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print("score : ", score) 

#random forest
#로그변환 전 score :  0.6845960053624471
#y만 로그변환 후 score :  0.6973102189874811
#x에 pop 변환 후 score : 0.6845960053624471
#둘다 로그변환 score :  0.6973102189874811


#linear regression
#로그변환 전 score :  0.6339063367757023
#y만 로그변환 score :  0.6561325856678322
#x에 pop 변환 후 score :  0.6341571218300018
#둘다 로그변환 score :  0.6558683309613873

