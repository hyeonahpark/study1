from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#1. data
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=334, train_size=0.9)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators' : 100,
    'learning_rate' : 0.1,
    'max_depth' : 5
}


#2. modeling
model = XGBRegressor(random_state = 334, **parameters)

model.set_params(gamma=0.4, learning_rate=0.2)

#3. compile
model.fit(x_train, y_train)

#4. predict
result = model.score(x_test, y_test)
print('사용 파라미터 : ', model.get_params())
print('model_score : ', result)