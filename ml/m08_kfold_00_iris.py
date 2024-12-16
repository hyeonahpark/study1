import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.svm import SVC

#1. data
x, y = load_iris(return_X_y=True) # 바로x와 y로 들어감 !

# print(x)
# dtc = DecisionTreeClassifier(random_state=156)
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. model
model = SVC()

#3. training
scores = cross_val_score(model, x, y, scoring='accuracy', cv=kfold)
print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

# print('교차 검증별 정확도:', np.round(scores,4))
# print('평균 검증 정확도:', np.round(np.mean(scores)))