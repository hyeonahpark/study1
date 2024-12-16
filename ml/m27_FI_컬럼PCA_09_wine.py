##############################
## 성능 좋은 것을 기준으로 해당 컬럼을 PCA로 만든 후 합치기

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#1. data
datasets= load_wine()

x = datasets.data
y = datasets.target
# y = pd.DataFrame(data=datasets.target)


from sklearn.model_selection import train_test_split
random_state=1223

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state)

#2. model
model = XGBClassifier(random_state=random_state)

model.fit(x_train, y_train)
print("===================", model.__class__.__name__, "====================")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

len = len(datasets.feature_names)
cut = round(len*0.2) ## 하위 20% 컬럼 갯수
print("하위 20% 컬럼 갯수 :", cut)
print(datasets.feature_names)
print('컬럼 제거 전 acc :', model.score(x_test, y_test))

percent = np.percentile(model.feature_importances_, 50)

rm_index=[]

for index, importance in enumerate(model.feature_importances_):
    if importance<=percent :
        rm_index.append(index)

#구린거
x_train1 = []     
for i in rm_index : 
    x_train1.append(x_train[:,i])
x_train1 = np.array(x_train1).T

x_test1 = []     
for i in rm_index : 
    x_test1.append(x_test[:,i])
x_test1 = np.array(x_test1).T

# 구린거 삭제한거
x_train = np.delete(x_train, rm_index, axis=1)
x_test = np.delete(x_test, rm_index, axis=1)

#구린거 1개로 합체
pca = PCA(n_components = 1)
x_train1 = pca.fit_transform(x_train1)
x_test1 = pca.transform(x_test1) 

#구린거+구린거 삭제한거
x_train = np.concatenate((x_train,x_train1),axis=1)
x_test = np.concatenate((x_test, x_test1),axis=1)


model = XGBClassifier(random_state=random_state)

model.fit(x_train, y_train)
print('PCA하고 합친 acc :', model.score(x_test, y_test))


# 컬럼 제거 전 acc : 1.0
# PCA하고 합친 acc : 1.0