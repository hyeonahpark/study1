import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#1. data
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8,
    stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. model
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

#model = XGBClassifier()
model = VotingClassifier(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    voting = 'soft',
    #voting = 'hard', #default
)

#3. training
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_y_pre : ', acc )

#xgb 
#최종점수 :  0.9649122807017544 
#acc_y_pre :  0.9649122807017544


#soft
# 최종점수 :  0.956140350877193
# acc_y_pre :  0.956140350877193

#hard
#최종점수 :  0.9649122807017544
#acc_y_pre :  0.9649122807017544