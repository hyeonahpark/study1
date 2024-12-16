# Pseudo Labeling 기법 : 모델 돌려서 나온 결과로 결측치를 찾아
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
from sklearn.ensemble import StackingRegressor

#1. data
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1199, train_size=0.8,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. model
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

train_list = []
test_list = []

model = StackingRegressor(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator = CatBoostRegressor(verbose=0),
    
    n_jobs = -1, 
    cv = 5
)

#3. 훈련
model.fit(x_train, y_train)

#4. predict
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 acc : ', r2_score(y_test, y_pred))

# model.score :  0.43641787376560925
# 스태킹 acc :  0.43641787376560925