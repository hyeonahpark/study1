### 회귀 ###

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

#1. data
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)

from sklearn.model_selection import train_test_split


random_state=5

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

#2. model
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models :
    model.fit(x_train, y_train)
    print("===================", model.__class__.__name__, "====================")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)
    

# =================== DecisionTreeRegressor ====================
# acc : -0.09901516042459213
# [0.05134564 0.00681589 0.36831101 0.11001809 0.05059884 0.054656
#  0.06505086 0.01789264 0.17766099 0.09765004]
# =================== RandomForestRegressor ====================
# acc : 0.526627803806025
# [0.06581766 0.01132514 0.32176746 0.10268358 0.04509679 0.05108646
#  0.06061744 0.02513689 0.24881283 0.06765574]
# =================== GradientBoostingRegressor ====================
# acc : 0.5290702255768158
# [0.05163314 0.01250898 0.33798427 0.12265751 0.0229866  0.04851005
#  0.0572883  0.0070048  0.28520088 0.05422548]
# =================== XGBRegressor ====================
# acc : 0.42414966816154154
# [0.04036526 0.05170747 0.22432816 0.09565202 0.05266059 0.07118361
#  0.06385097 0.04109477 0.29345533 0.06570182]