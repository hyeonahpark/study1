########## pandas 로 바꿔서 컬럼 삭제 ###############
#pd.DataFrame
# 컬럼명 : datasets.feature_names 안에 있찌 ! !


#실습
#피쳐임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
#데이터셋 재구성 후
#기존 모델결과와 비교 !

from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np

#1. data
datasets= fetch_california_housing()

x = datasets.data
y = datasets.target
# y = pd.DataFrame(data=datasets.target)


from sklearn.model_selection import train_test_split
random_state=1223

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state)

#2. model
model = RandomForestRegressor(random_state=random_state)

model.fit(x_train, y_train)
print("===================", model.__class__.__name__, "====================")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

len = len(datasets.feature_names)
cut = round(len*0.2) ## 하위 20% 컬럼 갯수
print("하위 20% 컬럼 갯수 :", cut)
print(datasets.feature_names)
print('컬럼 제거 전 acc :', model.score(x_test, y_test))

percent = np.percentile(model.feature_importances_, 20)

rm_index=[]

for index, importance in enumerate(model.feature_importances_):
    if importance<=percent :
        rm_index.append(index)

x_train = np.delete(x_train, rm_index, axis=1)
x_test = np.delete(x_test, rm_index, axis=1)

model = RandomForestRegressor(random_state=random_state)

model.fit(x_train, y_train)
print('컬럼 제거 후 acc :', model.score(x_test, y_test))


# 컬럼 제거 전 acc : 0.8142625704395272
# 컬럼 제거 후 acc : 0.8186710265070825