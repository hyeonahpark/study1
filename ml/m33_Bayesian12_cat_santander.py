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


path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sample_submission=pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape) # (200000, 201)
# print(test_csv.shape) # (200000, 200)
# print(sample_submission.shape) # (200000, 1)

# ################# x와 y 분리 ######################
x=train_csv.drop(['target'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
y=train_csv['target']

# print(train_csv)
# train_csv.boxplot()
# plt.show()
# exit()

unique,counts=np.unique(y, return_counts=True)
# print(np.unique(y, return_counts=True)) #이렇게 작성해서 바로 출력해도 됨. 출력값 : (array([0, 1], dtype=int64), array([179902,  20098], dtype=int64))

# print("고유한 요소:", unique) #고유한 요소: [0 1]
# print("각 요소의 개수:", counts) #각 요소의 개수: [179902  20098]

# print(pd.Series(y).value_counts)
# print(pd.value_counts(y)) 

x[x<0] = np.NaN
################## 결측치 확인 #####################
# print(train_csv.isna().sum())
# print(train_csv.isnull().sum())
# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())


############################X 데이터 로그변환##########################################
x['var_45'] = np.log1p(x['var_45'])
x['var_74'] = np.log1p(x['var_74'])
x['var_117'] = np.log1p(x['var_117'])
x['var_120'] = np.log1p(x['var_120'])

random_state=1186
x_train, x_test, y_train, y_test = train_test_split (x, y, random_state=random_state, train_size=0.9)

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

# # 최적 튠 ACC:  0.903048974639209
#  평균 ACC :  0.9145 (kfold)

# =================================================================================================================================================
# {'target': 0.9051, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 220.45664323952997, 'max_depth': 7.050909420819697, 'min_child_samples': 178.17852444702658, 'min_child_weight': 17.889473338694884, 'num_leaves': 25.21425778832579, 'reg_alpha': 16.011596541818474, 'reg_lambda': 6.167801123720769, 'subsample': 0.720167482946851}}
# 100 번 걸린시간 : 665.86 초

# {'target': 0.907, 'params': {'bagging_temperature': 0.19052851742414967, 'border_count': 207.0594490648213, 'depth': 10.0, 'l2_leaf_reg': 4.6154695017099705, 'learning_rate': 0.1, 'random_strength': 1.0}}
# 100 번 걸린시간 : 973.02 초