from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization
import time
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

x,y = load_digits(return_X_y=True) #return_X_y= True로 작성 가능

print(x)
print(y)
print(x.shape, y.shape) #(1797, 64) (1797,) 이미지면 (1797,8,8)

print(pd.value_counts(y,sort=False)) #sort=False를 하면 순서대로 정렬됨
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

# y=pd.get_dummies(y)
print(y.shape) #(1797, 10)

x=x/255.

#2. modeling
model=SVC()

#2. modeling
n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))


# loss :  0.16577006876468658
# ACC :  0.956

#dropout
# loss :  0.16080109775066376
# ACC :  0.972

# loss :  0.13260766863822937
# ACC :  0.961

#loss :  0.07087568938732147
# ACC :  0.983

#KFOLD
# ACC :  [0.98333333 0.98888889 0.98328691 0.99442897 0.98607242] 
#  평균 ACC :  0.9872

# ACC :  [0.98611111 0.99444444 0.98885794 0.98885794 0.98050139] 
#  평균 ACC :  0.9878