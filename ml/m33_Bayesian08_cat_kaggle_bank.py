import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import time 
from sklearn import base
import random as rn
import tensorflow as tf
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization


#1. data
path = 'C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)  # (165034, 13)
print(test_csv.shape)  # (110023,12)
print(sample_submission.shape) #(110023,1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

x=train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
test_csv=test_csv.drop(['CustomerId', 'Surname'], axis=1)

y=train_csv['Exited']

random_state=1186
x_train, x_test, y_train, y_test = train_test_split (x, y, random_state=random_state, train_size=0.9, stratify=y)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

#2. model
bayesian_params= {
    'learning_rate' : (0.001, 0.1),
    'depth' : (3, 10),
    'l2_leaf_reg' : (1, 10),
    'bagging_temperature' : (0, 5),
    'border_count' : (32, 256),
    'random_strength' : (1, 10)
}
cat_features = list(range(x_train.shape[1]))

def cat_boost(learning_rate, depth, l2_leaf_reg, bagging_temperature, border_count,
              random_strength):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)), #무조건 정수형 !
        'l2_leaf_reg' : int(round(l2_leaf_reg)),
        'bagging_temperature' :bagging_temperature,
        'border_count' : int(round(border_count)),
        'random_strength' : int(round(random_strength))
    }

    model = CatBoostClassifier(**params)

    model.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            #eval_metric='logloss',
            verbose=0,
            )

    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    return result


bay = BayesianOptimization(
    f = cat_boost,
    pbounds = bayesian_params,
    random_state = 333,
)

n_iter = 300
start_time = time.time()
bay.maximize(init_points = 5,
                   n_iter = n_iter)
end_time = time.time()


print(bay.max)
print(n_iter, '번 걸린시간 :', round(end_time-start_time,2), '초')
# ACC :  0.8617 (전)

# {'target': 0.8665777993213767, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 342.2182448225366, 'max_depth': 10.0, 'min_child_samples': 32.7979477834596, 'min_child_weight': 38.27444960407498, 'num_leaves': 24.0, 'reg_alpha': 15.453071221005638, 'reg_lambda': 0.6962822669879192, 'subsample': 1.0}}
# 300 번 걸린시간 : 175.56 초

# {'target': 0.8645176926805623, 'params': {'bagging_temperature': 4.633272065897993, 'border_count': 153.67548974407237, 'depth': 10.0, 'l2_leaf_reg': 1.7012120736766565, 'learning_rate': 0.1, 'random_strength': 5.51517385801236}}
# 300 번 걸린시간 : 317.81 초