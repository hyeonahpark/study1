import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

path = 'C:\\ai5\_data\\kaggle\\bike-sharing-demand\\' #슬래시 두개는 슬래시 하나로 인식함 (\a 와 \b는 문자열에서 특수문자로 인식하기 때문)
# path = 'C://ai5//_data//bike-sharing-demand//' #역슬래시로 작성해도 상관없음
# path = 'C:/ai5/_data/bike-sharing-demand/' 

train_csv=pd.read_csv(path + "train.csv", index_col=0)
test_csv=pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission=pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  #(10886, 11)
print(test_csv.shape)  #(6493, 8)
print(sampleSubmission.shape) # (6493, 1)

#casual, registered 는 미등록 사용자와 등록 사용자임. casual+registered 의 수는 count와 동일하므로 두 열을 삭제해도 됨.
print(train_csv.columns) #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    # 'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(train_csv.info()) #null 값 확인하기
print(test_csv.info())

print(train_csv.describe()) #count, mean, std, min, 1/4분위, 중위값, 3/4분위, max값 나옴. 어떤 주어진 값들을 크기의 순서대로 정렬했을 때 가장 중앙에 위치하는 값

################## 결측치 확인 #####################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

################# x와 y 분리 ######################

x=train_csv.drop(['count'], axis=1) #대괄호 하나 안에 다 넣기 ! 두개 이상은 리스트
y=train_csv['count']

# print(train_csv)
# train_csv.boxplot()
# plt.show()
#1. data

random_state=777
x_train, x_test, y_train, y_test = train_test_split (x, y, random_state=random_state, train_size=0.9)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

#2. model
bayesian_params= {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin':(9, 500),
    'reg_lambda':(-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)), #무조건 정수형 !
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' :int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin':max(int(round(max_bin)), 10),
        'reg_lambda':max(reg_lambda, 0),
        'reg_alpha' : reg_alpha
    }

    model = XGBRegressor(**params, n_jobs=-1)

    model.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            #eval_metric='logloss',
            verbose=0,
            )

    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    return result


bay = BayesianOptimization(
    f = xgb_hamsu,
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

# 최적 튠 ACC:  0.9993106638798117 (전)

# {'target': 0.9996975660324097, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 500.0, 'max_depth': 10.0, 'min_child_samples': 10.0, 'min_child_weight': 1.0, 'num_leaves': 40.0, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 0.8391829165226655}}
# 100 번 걸린시간 : 35.24 초