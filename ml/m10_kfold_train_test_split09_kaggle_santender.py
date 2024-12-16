import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold, cross_val_predict
import xgboost as xgb
from sklearn.metrics import accuracy_score

path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape) # (200000, 201)
# print(test_csv.shape) # (200000, 200)
# print(sample_submission.shape) # (200000, 1)

# ################# x와 y 분리 ######################
x=train_csv.drop(['target'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
y=train_csv['target']

# print(train_csv)
# train_csv.boxplot()
# plt.show()
# exit()

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True)) #이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1], dtype=int64), array([179902,  20098], dtype=int64))

# print("고유한 요소:", unique) #고유한 요소: [0 1]
# print("각 요소의 개수:", counts) #각 요소의 개수: [179902  20098]

# print(pd.Series(y).value_counts)
# print(pd.value_counts(y)) 

x[x<0] = np.NaN
################## 결측치 확인 #####################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())


############################X 데이터 로그변환##########################################
x['var_45'] = np.log1p(x['var_45'])
x['var_74'] = np.log1p(x['var_74'])
x['var_117'] = np.log1p(x['var_117'])
x['var_120'] = np.log1p(x['var_120'])
############################X 데이터 로그변환##########################################

# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1186)


##################### y 로그 변환####################################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
##################### y 로그 변환####################################
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

#2. model 랜덤 포레스트 모델 학습
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    random_state=1186,
    use_label_encoder=False,
    eval_metric='mlogloss',
    gpu_id=0
)

# fig, ax = plt.subplots(figsize=(10,12))
# plot_importance(xgb_model, ax=ax)
# model.fit(x_train, y_train, verbose=1)
# y_pred = model.predict(x_val)

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

#KFOLD
# ACC :  [0.916    0.91195  0.91715  0.911575 0.91575 ] 
#  평균 ACC :  0.9145

# StratifiedKFold
# ACC :  [0.91385 0.91325 0.9156  0.9145  0.9142 ] 
#  평균 ACC :  0.9143
# cross_val_predict ACC :  0.9039


