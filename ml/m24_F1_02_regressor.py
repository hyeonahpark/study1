#02_california
#03_diabetes


### 요 파일에 이 2개의 데이터 셋 다 넣어서 23번처럼 맹글기

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


#1. data
datasets1 = fetch_california_housing()
datasets2 = load_diabetes()

data = [datasets1, datasets2]
name = ['california', 'diabetes']
# x = data.data
# y = data.target

from sklearn.model_selection import train_test_split

random_state=5

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

#2. model
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]
i=0
for data in data:
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("===================", name[i] , "====================")
    for model in models :
        model.fit(x_train, y_train)
        print("===================", model.__class__.__name__, "====================")
        print('acc :', model.score(x_test, y_test))
        print(model.feature_importances_)
    i = i+1    
    
    
"""   
=================== california ====================
=================== DecisionTreeRegressor ====================
acc : 0.626398383963194
[0.51932888 0.04810277 0.05023234 0.02691352 0.03656326 0.12875284
 0.09317505 0.09693133]
=================== RandomForestRegressor ====================
acc : 0.8205430092435043
[0.51867961 0.05285011 0.04781134 0.02989032 0.03275855 0.13839308
 0.08919529 0.09042169]
=================== GradientBoostingRegressor ====================
acc : 0.794929544342801
[0.59381133 0.03162295 0.02014435 0.00446077 0.00251706 0.12735366
 0.10608471 0.11400517]
=================== XGBRegressor ====================
acc : 0.8428134619403308
[0.49298912 0.0616372  0.04621191 0.02440391 0.02217408 0.14406325
 0.09493851 0.11358204]
 
=================== diabetes ====================
=================== DecisionTreeRegressor ====================
acc : -0.0981028336866876
[0.05134564 0.00681589 0.36831101 0.11001809 0.05059884 0.054656
 0.06505086 0.01789264 0.17766099 0.09765004]
=================== RandomForestRegressor ====================
acc : 0.5219015437858573
[0.06581766 0.01132514 0.32176746 0.10268358 0.04509679 0.05108646
 0.06061744 0.02513689 0.24881283 0.06765574]
=================== GradientBoostingRegressor ====================
acc : 0.5281849235385081
[0.05163314 0.01250898 0.33798427 0.12265751 0.0229866  0.04851005
 0.0572883  0.0070048  0.28520088 0.05422548]
=================== XGBRegressor ====================
acc : 0.42414966816154154
[0.04036526 0.05170747 0.22432816 0.09565202 0.05266059 0.07118361
 0.06385097 0.04109477 0.29345533 0.06570182]
"""