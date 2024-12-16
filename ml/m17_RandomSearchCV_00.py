import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

#1. data
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test =train_test_split(
    x, y, shuffle=True, random_state=3333, train_size=0.8,
    stratify=y
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear', 'sigmoid'], 'degree' :[3, 4, 5]}, #24
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001,0.0001],}, #6
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.01, 0.001, 0.0001], 'degree' :[3, 4]} #24
    #총 54번
]

#2. model
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)

model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, n_iter=9, random_state=3333) #n_iter=10이 디폴트

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)  # 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_) # 최적의 파라미터 :  {'kernel': 'linear', 'degree': 3, 'C': 1}
print("best_score : ", model.best_score_)  # best_score :  0.9833333333333334
print("model.score : ", model.score(x_test, y_test)) # model.score :  1.0

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict)) # accuracy_score :  1.0


y_pred_best = model.best_estimator_.predict(x_test) 
print("최적 튠 ACC: ", accuracy_score(y_test, y_pred_best)) # 최적 튠 ACC:  1.0
print("시간 : ", round(end_time-start_time, 2))  # 시간 :  6.53

