import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import xgboost as xgb

#1. data
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test =train_test_split(
    x, y, shuffle=True, random_state=3333, train_size=0.8, stratify=y
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear', 'sigmoid'], 'degree' :[3, 4, 5], 'learning_rate':[0.1, 0.01, 0.001, 0.5, 0.05, 0.005]}, #144
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.1, 0.01, 0.001, 0.0001], 'learning_rate':[0.1, 0.01, 0.001]}, #12
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.01, 0.001, 0.0001], 'degree' :[3, 4], 'learning_rate':[0.5, 0.05, 0.005]} #24
    #총 54번
]

#2. model
model = RandomizedSearchCV(xgb.XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=3333)
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_) 
#최적의 매개변수 :  XGBClassifier(C=1, base_score=None, booster=None, callbacks=None,
            #   colsample_bylevel=None, colsample_bynode=None,
            #   colsample_bytree=None, degree=3, device=None,
            #   early_stopping_rounds=None, enable_categorical=False,
            #   eval_metric=None, feature_types=None, gamma=None,
            #   grow_policy=None, importance_type=None,
            #   interaction_constraints=None, kernel='linear', learning_rate=0.05,
            #   max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,
            #   max_delta_step=None, max_depth=None, max_leaves=None,
            #   min_child_weight=None, missing=nan, monotone_constraints=None,
            #   multi_strategy=None, n_estimators=None, ...)
            
print("최적의 파라미터 : ", model.best_params_) # 최적의 파라미터 :  {'learning_rate': 0.05, 'kernel': 'linear', 'degree': 3, 'C': 1}
print("best_score : ", model.best_score_) # best_score :  0.95
print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9333333333333333

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict)) #accuracy_score :  0.9333333333333333


y_pred_best = model.best_estimator_.predict(x_test) 
print("최적 튠 ACC: ", accuracy_score(y_test, y_pred_best)) # 최적 튠 ACC:  0.9333333333333333
print("시간 : ", round(end_time-start_time, 2)) # 시간 :  3.08

import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))
print(pd.DataFrame(model.cv_results_).columns)
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
#        'split0_test_score', 'split1_test_score', 'split2_test_score',
#        'split3_test_score', 'split4_test_score', 'mean_test_score',
#        'std_test_score', 'rank_test_score'],
#       dtype='object')
path = './_save/m15_GS_CV_01/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path+ 'm17_RS_cv_results.csv')