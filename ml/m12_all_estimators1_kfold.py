import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_validate, KFold, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')

#1. data
x, y = load_iris(return_X_y=True) # 바로x와 y로 들어감 !

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

print('sk 버전: ', sk.__version__) # 1.5.1
print('all Algorithm : ', all)
print('모델의 갯수 : ', len(all)) #43(classifier) , 55(regressor)

for name, model in all:
    try :
        #2. model
        model = model()
        #3. training
        model.fit(x_train, y_train)
        n_splits=5
        # kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1186)
        #3. training
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('=============', name, '================')
        print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
        y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predcit)
        print('cross_val_predict ACC : ', acc)
        #4. predict
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 : ', acc)
        
    except:
        print(name, '은 바보 멍충이!!!')
        
        
# ============= AdaBoostClassifier ================
# ACC :  [0.95652174 1.         0.95454545 0.95454545 0.95454545] 
#  평균 ACC :  0.964
# cross_val_predict ACC :  0.9736842105263158
# AdaBoostClassifier 의 정답률 :  0.9210526315789473
# ============= BaggingClassifier ================
# ACC :  [1.         1.         0.95454545 0.95454545 0.95454545] 
#  평균 ACC :  0.9727
# cross_val_predict ACC :  0.9736842105263158
# BaggingClassifier 의 정답률 :  0.9473684210526315
# ============= BernoulliNB ================
# ACC :  [0.86956522 0.82608696 0.68181818 0.68181818 0.77272727]
#  평균 ACC :  0.7664
# cross_val_predict ACC :  0.5263157894736842
# BernoulliNB 의 정답률 :  0.7105263157894737
# ============= CalibratedClassifierCV ================
# ACC :  [0.82608696 0.91304348 0.81818182 0.86363636 0.95454545] 
#  평균 ACC :  0.8751
# cross_val_predict ACC :  0.8947368421052632
# CalibratedClassifierCV 의 정답률 :  0.7894736842105263
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ComplementNB 은 바보 멍충이!!!
# ============= DecisionTreeClassifier ================
# ACC :  [1.         1.         1.         0.95454545 0.95454545]
#  평균 ACC :  0.9818
# cross_val_predict ACC :  0.9736842105263158
# DecisionTreeClassifier 의 정답률 :  0.9473684210526315
# ============= DummyClassifier ================
# ACC :  [0.34782609 0.34782609 0.31818182 0.36363636 0.36363636]
#  평균 ACC :  0.3482
# cross_val_predict ACC :  0.39473684210526316
# DummyClassifier 의 정답률 :  0.2894736842105263
# ============= ExtraTreeClassifier ================
# ACC :  [0.91304348 0.95652174 0.81818182 0.95454545 0.95454545]
#  평균 ACC :  0.9194
# cross_val_predict ACC :  0.9210526315789473
# ExtraTreeClassifier 의 정답률 :  0.9210526315789473
# ============= ExtraTreesClassifier ================
# ACC :  [0.95652174 0.95652174 1.         0.90909091 0.95454545] 
#  평균 ACC :  0.9553
# cross_val_predict ACC :  0.9736842105263158
# ExtraTreesClassifier 의 정답률 :  0.9473684210526315
# ============= GaussianNB ================
# ACC :  [0.95652174 0.95652174 0.90909091 0.90909091 1.        ]
#  평균 ACC :  0.9462
# cross_val_predict ACC :  0.9473684210526315
# GaussianNB 의 정답률 :  0.9473684210526315
# ============= GaussianProcessClassifier ================
# ACC :  [0.95652174 0.95652174 0.90909091 0.90909091 1.        ] 
#  평균 ACC :  0.9462
# cross_val_predict ACC :  0.9210526315789473
# GaussianProcessClassifier 의 정답률 :  0.9473684210526315
# ============= GradientBoostingClassifier ================
# ACC :  [1.         0.95652174 0.95454545 0.90909091 1.        ] 
#  평균 ACC :  0.964
# cross_val_predict ACC :  0.9736842105263158
# GradientBoostingClassifier 의 정답률 :  0.9473684210526315
# ============= HistGradientBoostingClassifier ================
# ACC :  [1.         1.         1.         0.90909091 0.95454545] 
#  평균 ACC :  0.9727
# cross_val_predict ACC :  0.39473684210526316
# HistGradientBoostingClassifier 의 정답률 :  0.9473684210526315
# ============= KNeighborsClassifier ================
# ACC :  [0.95652174 0.95652174 0.90909091 0.90909091 0.90909091]
#  평균 ACC :  0.9281
# cross_val_predict ACC :  0.9210526315789473
# KNeighborsClassifier 의 정답률 :  0.9473684210526315
# ============= LabelPropagation ================
# ACC :  [0.95652174 0.95652174 0.95454545 0.90909091 0.95454545]
#  평균 ACC :  0.9462
# cross_val_predict ACC :  0.9210526315789473
# LabelPropagation 의 정답률 :  0.9736842105263158
# ============= LabelSpreading ================
# ACC :  [0.95652174 0.95652174 0.95454545 0.90909091 0.95454545]
#  평균 ACC :  0.9462
# cross_val_predict ACC :  0.9210526315789473
# LabelSpreading 의 정답률 :  0.9736842105263158
# ============= LinearDiscriminantAnalysis ================
# ACC :  [0.95652174 1.         1.         0.95454545 1.        ]
#  평균 ACC :  0.9822
# cross_val_predict ACC :  0.9473684210526315
# LinearDiscriminantAnalysis 의 정답률 :  0.9736842105263158
# ============= LinearSVC ================
# ACC :  [0.82608696 0.95652174 0.90909091 0.90909091 1.        ]
#  평균 ACC :  0.9202
# cross_val_predict ACC :  0.9736842105263158
# LinearSVC 의 정답률 :  0.9210526315789473
# ============= LogisticRegression ================
# ACC :  [0.95652174 0.95652174 0.95454545 0.90909091 1.        ] 
#  평균 ACC :  0.9553
# cross_val_predict ACC :  0.9210526315789473
# LogisticRegression 의 정답률 :  0.9473684210526315
# ============= LogisticRegressionCV ================
# ACC :  [0.95652174 1.         1.         0.95454545 0.90909091] 
#  평균 ACC :  0.964
# cross_val_predict ACC :  0.9736842105263158
# LogisticRegressionCV 의 정답률 :  0.9473684210526315
# ============= MLPClassifier ================
# ACC :  [0.95652174 0.95652174 0.90909091 0.90909091 0.95454545] 
#  평균 ACC :  0.9372
# cross_val_predict ACC :  0.9210526315789473
# MLPClassifier 의 정답률 :  0.9473684210526315
# MultiOutputClassifier 은 바보 멍충이!!!
# MultinomialNB 은 바보 멍충이!!!
# ============= NearestCentroid ================
# ACC :  [0.95652174 0.91304348 0.77272727 0.81818182 0.90909091]
#  평균 ACC :  0.8739
# cross_val_predict ACC :  0.8947368421052632
# NearestCentroid 의 정답률 :  0.8421052631578947
# ============= NuSVC ================
# ACC :  [0.95652174 0.95652174 0.95454545 0.90909091 1.        ]
#  평균 ACC :  0.9553
# cross_val_predict ACC :  0.9736842105263158
# NuSVC 의 정답률 :  0.9736842105263158
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# ============= PassiveAggressiveClassifier ================
# ACC :  [0.95652174 0.95652174 0.86363636 0.86363636 1.        ]
#  평균 ACC :  0.9281
# cross_val_predict ACC :  0.8157894736842105
# PassiveAggressiveClassifier 의 정답률 :  0.8947368421052632
# ============= Perceptron ================
# ACC :  [0.91304348 0.95652174 0.90909091 0.77272727 0.63636364]
#  평균 ACC :  0.8375
# cross_val_predict ACC :  0.8157894736842105
# Perceptron 의 정답률 :  0.7894736842105263
# ============= QuadraticDiscriminantAnalysis ================
# ACC :  [0.95652174 1.         1.         0.95454545 0.90909091]
#  평균 ACC :  0.964
# cross_val_predict ACC :  0.9736842105263158
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9736842105263158
# ============= RadiusNeighborsClassifier ================
# ACC :  [       nan 0.95652174 0.86363636 0.90909091 1.        ]
#  평균 ACC :  nan
# cross_val_predict ACC :  0.868421052631579
# RadiusNeighborsClassifier 의 정답률 :  0.9473684210526315
# ============= RandomForestClassifier ================
# ACC :  [1.         0.95652174 0.95454545 0.90909091 0.95454545] 
#  평균 ACC :  0.9549
# cross_val_predict ACC :  0.9736842105263158
# RandomForestClassifier 의 정답률 :  0.9473684210526315
# ============= RidgeClassifier ================
# ACC :  [0.7826087  0.86956522 0.81818182 0.86363636 0.95454545]
#  평균 ACC :  0.8577
# cross_val_predict ACC :  0.7894736842105263
# RidgeClassifier 의 정답률 :  0.7105263157894737
# ============= RidgeClassifierCV ================
# ACC :  [0.7826087  0.86956522 0.81818182 0.86363636 0.95454545]
#  평균 ACC :  0.8577
# cross_val_predict ACC :  0.7894736842105263
# RidgeClassifierCV 의 정답률 :  0.7105263157894737
# ============= SGDClassifier ================
# ACC :  [0.82608696 0.82608696 0.95454545 0.77272727 1.        ]
#  평균 ACC :  0.8759
# cross_val_predict ACC :  0.8157894736842105
# SGDClassifier 의 정답률 :  0.9210526315789473
# ============= SVC ================
# ACC :  [0.95652174 0.95652174 0.95454545 0.90909091 0.95454545]
#  평균 ACC :  0.9462
# cross_val_predict ACC :  0.9473684210526315
# SVC 의 정답률 :  0.9473684210526315
# StackingClassifier 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!