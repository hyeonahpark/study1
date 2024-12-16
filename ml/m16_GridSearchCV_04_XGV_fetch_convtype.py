import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import time
import xgboost as xgb
import pandas as pd

#1. data

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# y=pd.get_dummies(y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test =train_test_split(
    x, y, shuffle=True, random_state=3333, train_size=0.8
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist']}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #4
] #48

#2. model
model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)
start_time = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True) #이렇게 하면 x_test, y_test가 validation data가 됨
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_) 
print("최적의 파라미터 : ", model.best_params_)
print("best_score : ", model.best_score_) 
print("model.score : ", model.score(x_test, y_test)) 

y_predict = model.predict(x_test)
y_predict = le.inverse_transform(y_predict)
print('accuracy_score : ', accuracy_score(y_test, y_predict)) 


y_pred_best = model.best_estimator_.predict(x_test) 
y_pred_best = le.inverse_transform(y_pred_best)
print("최적 튠 ACC: ", accuracy_score(y_test, y_pred_best)) 
print("시간 : ", round(end_time-start_time, 2)) 

# model.score :  0.9726771253754206
# accuracy_score :  0.9726771253754206
# 최적 튠 ACC:  0.9726771253754206
# 시간 :  2484.03