from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, KFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score

#1. data
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# print(df)

df['target'] = datasets.target

x = df.drop(['target'], axis=1).copy()
y = df['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

#2. model
model = RandomForestRegressor(random_state= 1186, max_depth=5, min_samples_split=3)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = r2_score(y_test, y_predcit)
print('cross_val_predict ACC : ', acc)

# ACC :  [0.64992406 0.66348391 0.64540454 0.67241787 0.66772374] 
#  평균 ACC :  0.6598


# ACC :  [0.64992406 0.66348391 0.64540454 0.67241787 0.66772374] 
#  평균 ACC :  0.6598
# cross_val_predict ACC :  0.6689965631201347