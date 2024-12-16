import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import PolynomialFeatures

#1. data
x, y = load_digits(return_X_y=True)

pf = PolynomialFeatures(degree = 2, include_bias=False)
x = pf.fit_transform(x)
random_state=4123
x_train, x_test, y_train, y_test = train_test_split (x, y, random_state=random_state, train_size=0.9, stratify=y)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

#2. model
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(task_type='GPU', devices='0', verbose=0)

train_list = []
test_list = []

model = StackingClassifier(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator = CatBoostClassifier(task_type='GPU', devices='0', verbose=0),
    #n_jobs = -1, 
    cv = 5
)

#3. 훈련
model.fit(x_train, y_train)

#4. predict
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 acc : ', r2_score(y_test, y_pred))

# model.score :  0.9944444444444445
# 스태킹 acc :  0.9831649831649831