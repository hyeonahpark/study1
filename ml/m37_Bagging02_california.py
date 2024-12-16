import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

#1. data
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. model
#model = DecisionTreeRegressor()
#model = LinearRegression()
model = RandomForestRegressor()
# model = BaggingRegressor(RandomForestRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=False) #디폴트, 중복허용
#3. training
model.fit(x_train, y_train)

#4. predict
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('acc_score :', acc)

#Decision
# 최종점수 :  0.6101969128750683
# acc_score : 0.6101969128750683

#Decision Bagging, 부트스트랩 트루
# 최종점수 :  0.8107197085695325
# acc_score : 0.8107197085695325

#Decision Bagging, 부트스트랩 false
# 최종점수 :  0.6302361114247275
# acc_score : 0.6302361114247275

#LinearRegreesion
# 최종점수 :  0.6011614863584488
# acc_score : 0.6011614863584488

#LinearRegression bagging, 부트스트랩 트루
# 최종점수 :  0.5962043412746386
# acc_score : 0.5962043412746386

#LinearRegression bagging, 부트스트랩 false
# 최종점수 :  0.6011614863584488
# acc_score : 0.6011614863584488

#XGB
# 최종점수 :  0.834965482631102
# acc_score : 0.834965482631102

#XGB 배깅, 부트스트랩 트루
# 최종점수 :  0.852804584098082
# acc_score : 0.852804584098082

#XGB 배깅, 부트스트랩 false
# 최종점수 :  0.8349654781075344
# acc_score : 0.8349654781075344

#Randomforest

#Randomforest 배깅, 부트스트랩 트루
# 최종점수 :  0.8075877878538014
# acc_score : 0.8075877878538014

#Randomforest 배깅, 부트스트랩 false
# 최종점수 :  0.8145608091094456
# acc_score : 0.8145608091094456