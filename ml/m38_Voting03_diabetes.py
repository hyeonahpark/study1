import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

#1. data
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. model
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

#model = XGBRegressor()
model = VotingRegressor(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    #voting = 'soft',
    #voting = 'hard', #default
)

#3. training
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('acc_y_pre : ', acc )

#xgb 
#최종점수 :  0.3505316025455074
#acc_y_pre :  0.3505316025455074


#voting
#최종점수 :  0.4618399590233273
#acc_y_pre :  0.4618399590233273