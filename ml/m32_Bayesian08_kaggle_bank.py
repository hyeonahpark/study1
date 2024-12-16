import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
from xgboost import XGBClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


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

    model = XGBClassifier(**params, n_jobs=-1)

    model.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            #eval_metric='logloss',
            verbose=0,
            )

    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    return result


bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds = bayesian_params,
    random_state = 42,
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
