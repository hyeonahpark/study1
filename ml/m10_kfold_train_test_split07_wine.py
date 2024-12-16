from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)

# print(y)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y)) 
#1    71
#0    59
#2    48

# y=pd.get_dummies(y)
# print(y.shape) #(178, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=MinMaxScaler()
# scaler=MinMaxScaler()
scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

#2. modeling
model=RandomForestClassifier(random_state= 1186, max_depth=5, min_samples_split=3)

#2. modeling
n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))


y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predcit)
print('cross_val_predict ACC : ', acc)

#loss :  0.16457915306091309
# ACC :  0.944

# loss :  0.16457915306091309
# ACC :  0.944


#KFOLD
# ACC :  [0.97222222 0.97222222 1.         1.         0.97142857] 
#  평균 ACC :  0.9832

# StratifiedKFold
# ACC :  [1.         0.97222222 0.97222222 1.         0.97142857] 
#  평균 ACC :  0.9832

# cross_val_predict ACC :  0.9333333333333333