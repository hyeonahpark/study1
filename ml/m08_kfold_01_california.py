from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.svm import SVC

#1. data
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# print(df)

df['target'] = datasets.target

x = df.drop(['target'], axis=1).copy()
y = df['target']

#2. model
model = RandomForestRegressor(random_state= 1186, max_depth=5, min_samples_split=3)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))


# ACC :  [0.64992406 0.66348391 0.64540454 0.67241787 0.66772374] 
#  평균 ACC :  0.6598