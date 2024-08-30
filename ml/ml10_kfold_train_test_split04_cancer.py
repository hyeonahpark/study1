from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold, cross_val_predict
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

#[실습]
#R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)

#2. modeling
model=SVC()
#3. fit
# model.fit(x_train, y_train)
n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))


y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predcit)
print('cross_val_predict ACC : ', acc)

# KFOLD
# ACC :  [0.92105263 0.87719298 0.90350877 0.94736842 0.91150442] 
#  평균 ACC :  0.9121


#StratifiedKFold
# ACC :  [0.92105263 0.93859649 0.92105263 0.92982456 0.86725664] 
#  평균 ACC :  0.9156

# ACC :  [0.92105263 0.93859649 0.92105263 0.92982456 0.86725664] 
#  평균 ACC :  0.9156
# cross_val_predict ACC :  0.8947368421052632