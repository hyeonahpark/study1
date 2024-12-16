#https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC, SVR
import xgboost as xgb
from sklearn.metrics import accuracy_score

path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path +"sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  # (61878, 94)
# print(test_csv.shape)  # (144368, 93)
# print(sampleSubmission_csv.shape) # (144368, 9)

################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

################# x와 y 분리 ######################
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])

x=train_csv.drop(['target'], axis=1)
y=train_csv['target']

# print(x)

# print(x.shape) # (61878, 93)
# print(y.shape) # (61878,)

# print(y)

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True) 이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1]), array([212, 357], dtype=int64))

# print("고유한 요소:", unique) #고유한 요소: [0 1 2 3 4 5 6 7 8]
# print("각 요소의 개수:", counts) #각 요소의 개수: [ 1929 16122  8004  2691  2739 14135  2839  8464  4955]

# y=pd.get_dummies(y)
# print(y.shape) #(61878, 9)

x=x.to_numpy()
x=x/255.
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

#2. modeling
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    random_state=1186,
    use_label_encoder=False,
    eval_metric='mlogloss',
    gpu_id=0
)
#2. modeling
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predcit)
print('cross_val_predict ACC : ', acc)

# loss :  0.6233358979225159
# ACC :  0.783

# loss :  0.5623487234115601
# ACC :  0.78

#cnn
# loss :  0.561005711555481
# ACC :  0.787

#KFOLD
# ACC :  [0.81916613 0.82393342 0.81148998 0.81745455 0.81632323] 
#  평균 ACC :  0.8177

# StratifiedKFold
# ACC :  [0.81948933 0.82005495 0.81827731 0.81381818 0.8150303 ] 
#  평균 ACC :  0.8173

# cross_val_predict ACC :  0.7872656755009696