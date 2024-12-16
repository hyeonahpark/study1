import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor

#1. data
x, y = load_linnerud(return_X_y=True)
print(x.shape, y.shape) #(20, 3) (20, 3)
print(x)
print(y)
##############  요런 데이터 ################
#       x           y
# [5. 162. 60.] -> [191. 36. 50.]
# .....................
# [2. 110. 43.] -> [138. 33. 68.]

#2. model
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pred), 4)) #  3.7693
print(model.predict([[2,110,43]])) #[[155.84  34.4   63.12]]

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'Ridge 스코어 : ', round(mean_absolute_error(y, y_pred), 4)) #  7.4569
print(model.predict([[2,110,43]])) #[[187.32842123  37.0873515   55.40215097]]

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, 'XGB 스코어 : ', round(mean_absolute_error(y, y_pred), 4)) #  0.0008
print(model.predict([[2,110,43]])) #[[138.0005    33.002136  67.99897 ]]

#model = CatBoostRegressor() # 에러
#model.fit(x, y)
#y_pred = model.predict(x)
#print(model.__class__.__name__, 'cat 스코어 : ', round(mean_absolute_error(y, y_pred), 4)) #  7.4569
#print(model.predict([[2,110,43]])) #[[155.84  34.4   63.12]]


from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

model = MultiOutputRegressor(LGBMRegressor())   
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # MultiOutputRegressor 스코어 :  8.91
print(model.predict([[2,110,43]]))  # [[178.6  35.4  56.1]]

model = MultiOutputRegressor(CatBoostRegressor())      
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # MultiOutputRegressor 스코어 :  0.2154
print(model.predict([[2,110,43]]))  # [[138.97756017  33.09066774  67.61547996]]


#### catboost ####
model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # CatBoostRegressor 스코어 :  0.0638
print(model.predict([[2,110,43]]))  # [[138.21649371  32.99740595  67.8741709 ]]



