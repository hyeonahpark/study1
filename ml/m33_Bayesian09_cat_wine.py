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
from sklearn.datasets import load_wine


#1. data
x, y = load_wine(return_X_y=True)

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

n_iter = 100
start_time = time.time()
bay.maximize(init_points = 5,
                   n_iter = n_iter)
end_time = time.time()


print(bay.max)
print(n_iter, '번 걸린시간 :', round(end_time-start_time,2), '초')

# =================================================
# 컬럼 제거 후 acc : 1.0 (전)

# {'target': 1.0, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 439.4626640712955, 'max_depth': 3.0, 'min_child_samples': 53.59853198342551, 'min_child_weight': 1.0, 'num_leaves': 25.413403952754955, 'reg_alpha': 5.783989953392123, 'reg_lambda': 0.3840664122795468, 'subsample': 0.5}}
# 100 번 걸린시간 : 16.97 초

# {'target': 1.0, 'params': {'bagging_temperature': 3.3742014913624736, 'border_count': 255.99370836154415, 'depth': 8.768885710694905, 'l2_leaf_reg': 1.9676452434975131, 'learning_rate': 0.07726586078481622, 'random_strength': 9.921547400933063}}
# 100 번 걸린시간 : 57.48 초