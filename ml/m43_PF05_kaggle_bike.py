import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier
import pandas as pd 
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import PolynomialFeatures

path = 'C:/ai5/_data/kaggle/bike-sharing-demand/' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
# path = 'C://ai5//_data//bike-sharing-demand//' #역슬래시로 작성해도 상관없음
# path = 'C:/ai5/_data/bike-sharing-demand/' 

train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission=pd.read_csv(path + "sampleSubmission.csv", index_col=0)

################# x와 y 분리 ######################

x=train_csv.drop(['count'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
y=train_csv['count']

pf = PolynomialFeatures(degree = 2, include_bias=False)
x = pf.fit_transform(x)

random_state=1199
x_train, x_test, y_train, y_test = train_test_split (x, y, random_state=random_state, train_size=0.9)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

#2. model
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

train_list = []
test_list = []

model = StackingRegressor(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator = CatBoostRegressor(task_type='GPU', devices='0', verbose=0),
    
    n_jobs = -1, 
    cv = 5
)

#3. 훈련
model.fit(x_train, y_train)

#4. predict
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 acc : ', r2_score(y_test, y_pred))

# 원래
# best_score :  0.9993074988900975
# model.score :  0.9993106638798117


# -------------------PF ----------------
# model.score :  0.9991852195779456
# 스태킹 acc :  0.9991852195779456
