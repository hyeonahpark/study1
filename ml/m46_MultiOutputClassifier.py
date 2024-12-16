import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import StackingRegressor
import pandas as pd 

np.random.seed(777)

def create_multiclass_data_with_labels():
    #x 데이터 생성 (20,3)
    x = np.random.rand(20, 3)
    y = np.random.randint(0, 5, size=(20,3)) #각 클래스에 0부터 9까지의 값
    
    #데이터프레임으로 변환
    x_df = pd.DataFrame(x, columns = ['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns = ['label2', 'label2', 'label3'])
    
    return x_df.values, y_df.values

# 각 열 별로 accuracy 계산
def multioutput_accuracy(y_true, y_pred):
    accuracies = []
    for i in range(y_true.shape[1]):  # 각 열에 대해
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies.append(acc)
    return np.mean(accuracies)  # 평균 정확도 반환

x,y = create_multiclass_data_with_labels()
print(" X 데이터 :")
print(x)
print("\n y 데이터 :")
print(y)

#2. model
model = RandomForestClassifier()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'randomforest 스코어 : ', round(mean_absolute_error(y, y_pred), 4)) #  0.0
print(model.predict([[2,110,43]])) #[[3 0 4]]


model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'Ridge 스코어 : ', round(mean_absolute_error(y, y_pred), 4)) #  0.9461
print(model.predict([[2,110,43]])) #[[103.06874446  -5.27432632 151.90625879]]

#model = XGBClassifier() #에러
#model.fit(x, y)
#y_pred = model.predict(x)
#print(model.__class__.__name__, 'XGB 스코어 : ', round(mean_absolute_error(y, y_pred), 4)) #  0.0008
#print(model.predict([[2,110,43]])) #[[138.0005    33.002136  67.99897 ]]

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

model = MultiOutputClassifier(XGBClassifier())   
model.fit(x,y)
print(model.__class__.__name__, 'xgb 스코어 : ', round(mean_absolute_error(y, y_pred), 4))  #  0.9461
print(model.predict([[2, 110, 43]]))  #[[0 0 4]]


model = MultiOutputClassifier(LGBMClassifier())   
model.fit(x,y)
print(model.__class__.__name__, 'lgbm스코어 : ', round(mean_absolute_error(y, y_pred), 4))  #  0.9461
print(model.predict([[2, 110, 43]]))  # [[3 2 2]]



model = MultiOutputClassifier(CatBoostClassifier())      
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre.reshape(20,3)), 4))   # MultiOutputRegressor 스코어 :  0.0
print(model.predict([[2,110,43]]))  # [[[3 4 3]]]

#### catboost ####
#model = CatBoostClassifier(loss_function='MultiRMSE')
#model.fit(x,y)
#y_pre = model.predict(x)
#print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # CatBoostRegressor 스코어 :  0.0638
#print(model.predict([[2,110,43]]))  # [[138.21649371  32.99740595  67.8741709 ]]
