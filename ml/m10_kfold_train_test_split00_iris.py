import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import cross_val_score, cross_validate, KFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

#1. data
x, y = load_iris(return_X_y=True) # 바로x와 y로 들어감 !

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. model
model = SVC()

#3. training
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=kfold) # 기준 점수 확인
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predcit)
print(y_test)

acc = accuracy_score(y_test, y_predcit)
print('cross_val_predict ACC : ', acc) #cross_val_predict ACC :  0.9210526315789473

# print('교차 검증별 정확도:', np.round(scores,4))
# print('평균 검증 정확도:', np.round(np.mean(scores)))