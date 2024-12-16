# Pseudo Labeling 기법 : 모델 돌려서 나온 결과로 결측치를 찾아

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd 
from sklearn.ensemble import StackingClassifier

#1. data
path = 'C:\\ai5\\_data\\dacon\\diabetes\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

################# x와 y 분리 ######################
x=train_csv.drop(['Outcome'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트

y=train_csv['Outcome']

random_state = 1199
x_train, x_test, y_train, y_test = train_test_split (x, y, random_state=random_state, train_size=0.9, stratify=y)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)


#2. model
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

train_list = []
test_list = []

model = StackingClassifier(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator = CatBoostClassifier(verbose=0),
    n_jobs = -1, 
    cv = 5
)

#3. 훈련
model.fit(x_train, y_train)

#4. predict
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 acc : ', accuracy_score(y_test, y_pred))

# model.score :  0.7878787878787878
# 스태킹 acc :  0.7878787878787878