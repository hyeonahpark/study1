import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import time
import xgboost as xgb
import pandas as pd

#1. data

x,y = load_digits(return_X_y=True) #return_X_y= True로 작성 가능

x=x/255.

# y=pd.get_dummies(y)

x_train, x_test, y_train, y_test =train_test_split(
    x, y, shuffle=True, random_state=3333, train_size=0.8
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {'n_jobs':[-1], 'n_estimators': [100, 500], 'max_depth': [6, 10, 12],
     'min_samples_leaf':[3, 10]},
    {'n_jobs':[-1],  'max_depth': [6, 8, 10, 12],
     'min_samples_leaf':[3, 5, 7, 10]},
    {'n_jobs':[-1],   'min_samples_leaf':[3, 5, 7, 10], 'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1], 'min_samples_split': [2,3,5,10]}
]

#2. model
model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_) 
print("최적의 파라미터 : ", model.best_params_)
print("best_score : ", model.best_score_) 
print("model.score : ", model.score(x_test, y_test)) 

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict)) 


y_pred_best = model.best_estimator_.predict(x_test) 
print("최적 튠 ACC: ", accuracy_score(y_test, y_pred_best)) 
print("시간 : ", round(end_time-start_time, 2)) 

# 최적의 매개변수 :  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=None, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=6, max_leaves=None,
#               min_child_weight=None, min_samples_leaf=3, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 500, 'n_jobs': -1}
# best_score :  0.963811459543167
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667
# 최적 튠 ACC:  0.9666666666666667
# 시간 :  7.09
# 시간 :  7.76