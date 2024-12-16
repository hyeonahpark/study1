import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

#1. data
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. model
#model = DecisionTreeRegressor()
#model = XGBRegressor()
#model = RandomForestRegressor()
model = BaggingRegressor(XGBRegressor(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=4444,
                          bootstrap=False) #디폴트, 중복허용
#3. training
model.fit(x_train, y_train)

#4. predict
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('acc_score :', acc)


#XGB
# 최종점수 :  0.3505316025455074
# acc_score : 0.3505316025455074

#XGB 배깅, 부트스트랩 트루
# 최종점수 :  0.5058813301969447
# acc_score : 0.5058813301969447

#XGB 배깅, 부트스트랩 false
# 최종점수 :  0.35053165996075253
# acc_score : 0.35053165996075253

#Randomforest
# 최종점수 :  0.5169444866596844
# acc_score : 0.5169444866596844

#Randomforest 배깅, 부트스트랩 트루
# 최종점수 :  0.5287592525821849
# acc_score : 0.5287592525821849

#Randomforest 배깅, 부트스트랩 false
# 최종점수 :  0.49717253613568513
# acc_score : 0.49717253613568513