import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import time
import xgboost as xgb
import pandas as pd

#1. data
dataset = load_diabetes()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# print(df)
# df.boxplot()
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

x_train, x_test, y_train, y_test =train_test_split(
    x, y, shuffle=True, random_state=3333, train_size=0.8
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {'n_jobs':[-1], 'n_estimators': [100, 500], 'max_depth': [6, 10, 12],
     'min_samples_leaf':[3, 10]},
    {'n_jobs':[-1],  'max_depth': [6, 8, 10, 12],
     'min_samples_leaf':[3, 5, 7, 10]},
    {'n_jobs':[-1],   'min_samples_leaf':[3, 5, 7, 10], 'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1], 'min_samples_split': [2,3,5,10]}
]

#2. model
model = GridSearchCV(xgb.XGBRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_) #최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_) #최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
print("best_score : ", model.best_score_) #best_score :  0.9833333333333334
print("model.score : ", model.score(x_test, y_test)) #model.score :  1.0

y_predict = model.predict(x_test)
print('accuracy_score : ', r2_score(y_test, y_predict)) #accuracy_score :  1.0


y_pred_best = model.best_estimator_.predict(x_test) 
print("최적 튠 ACC: ", r2_score(y_test, y_pred_best)) #최적 튠 ACC:  1.0
print("시간 : ", round(end_time-start_time, 2)) #시간 :  1.5
