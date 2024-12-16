
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC, SVR
import xgboost as xgb
from sklearn.metrics import accuracy_score


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


from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])
test_csv[:] = scalar.fit_transform(test_csv[:])

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1186)

# gbrt = GradientBoostingClassifier(random_state=0)
# gbrt.fit(x_train, y_train)


#2. modeling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1186)


model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.005,
    max_depth=7,
    random_state=1186,
    use_label_encoder=False,
    eval_metric='mlogloss',
    gpu_id=0
)

# fig, ax = plt.subplots(figsize=(10,12))
# plot_importance(xgb_model, ax=ax)
# model.fit(x_train, y_train, verbose=1)
# y_pred = model.predict(x_val)

#2. modeling
n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#3. training
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predcit)
print('cross_val_predict ACC : ', acc)

#KFOLD
# ACC :  [0.86224134 0.86048414 0.86454388 0.85984791 0.8630552 ] 
#  평균 ACC :  0.862

# StratifiedKFold
# ACC :  [0.85957524 0.86421062 0.86109007 0.86308965 0.86044962] 
#  평균 ACC :  0.8617

# cross_val_predict ACC :  0.8578526417838099

