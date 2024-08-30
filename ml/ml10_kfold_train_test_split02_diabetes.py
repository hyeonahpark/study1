from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, KFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score



#1. data
dataset = load_diabetes()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# print(df)
# df.boxplot()
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

#[실습]
#R2 0.62 이상

# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)

##################### y 로그 변환####################################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
##################### y 로그 변환####################################
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

#2. model 랜덤 포레스트 모델 학습
model = RandomForestRegressor(random_state= 52151, max_depth=5, min_samples_split=3)

#3. fit
# model.fit(x_train, y_train)
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = r2_score(y_test, y_predcit)
print('cross_val_predict ACC : ', acc)


#변환 x  score :  0.5187753598635303
#변환    score :  0.5366610211847753


# ACC :  [0.39179123 0.37331273 0.51366394 0.37536275 0.45105447] 
#  평균 ACC :  0.421


# ACC :  [0.39179123 0.37331273 0.51366394 0.37536275 0.45105447] 
#  평균 ACC :  0.421
# cross_val_predict ACC :  0.21803821651636957