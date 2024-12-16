import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression

#1. data
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8,
    stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. model
#model = DecisionTreeClassifier()
#model = LogisticRegression()
model = BaggingClassifier(LogisticRegression(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=4444,
                          bootstrap=True) #디폴트, 중복허용
#3. training
model.fit(x_train, y_train)

#4. predict
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

#Decision
#최종점수 :  0.9473684210526315
# acc_score : 0.947368421052631

#Decision Bagging, 부트스트랩 트루
# 최종점수 :  0.9385964912280702
# acc_score : 0.9385964912280702

#로지스틱
# 최종점수 :  0.9736842105263158
# acc_score : 0.9736842105263158

#LogisticRegression 배깅, 부트스트랩 트루
# 최종점수 :  0.9649122807017544
# acc_score : 0.9649122807017544


