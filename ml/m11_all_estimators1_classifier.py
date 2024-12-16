import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_validate, KFold, cross_val_predict
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
        #4. predict
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 : ', acc)
        
    except:
        print(name, '은 바보 멍충이!!!')
        
        
#모델의 갯수 :  41
# AdaBoostClassifier 의 정답률 :  0.9210526315789473
# BaggingClassifier 의 정답률 :  0.9473684210526315
# BernoulliNB 의 정답률 :  0.7105263157894737
# CalibratedClassifierCV 의 정답률 :  0.7894736842105263
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ComplementNB 은 바보 멍충이!!!
# DecisionTreeClassifier 의 정답률 :  0.9473684210526315
# DummyClassifier 의 정답률 :  0.2894736842105263
# ExtraTreeClassifier 의 정답률 :  0.9210526315789473
# ExtraTreesClassifier 의 정답률 :  0.9473684210526315
# GaussianNB 의 정답률 :  0.9473684210526315
# GaussianProcessClassifier 의 정답률 :  0.9473684210526315
# GradientBoostingClassifier 의 정답률 :  0.9473684210526315
# HistGradientBoostingClassifier 의 정답률 :  0.9473684210526315
# KNeighborsClassifier 의 정답률 :  0.9473684210526315
# LabelPropagation 의 정답률 :  0.9736842105263158
# LabelSpreading 의 정답률 :  0.9736842105263158
# LinearDiscriminantAnalysis 의 정답률 :  0.9736842105263158
# LinearSVC 의 정답률 :  0.9210526315789473
# LogisticRegression 의 정답률 :  0.9473684210526315
# LogisticRegressionCV 의 정답률 :  0.9473684210526315
# MLPClassifier 의 정답률 :  0.9473684210526315
# MultiOutputClassifier 은 바보 멍충이!!!
# MultinomialNB 은 바보 멍충이!!!
# NearestCentroid 의 정답률 :  0.8421052631578947
# NuSVC 의 정답률 :  0.9736842105263158
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# PassiveAggressiveClassifier 의 정답률 :  0.7631578947368421
# Perceptron 의 정답률 :  0.7894736842105263
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9736842105263158
# RadiusNeighborsClassifier 의 정답률 :  0.9473684210526315
# RandomForestClassifier 의 정답률 :  0.9473684210526315
# RidgeClassifier 의 정답률 :  0.7105263157894737
# RidgeClassifierCV 의 정답률 :  0.7105263157894737
# SGDClassifier 의 정답률 :  0.6842105263157895
# SVC 의 정답률 :  0.9473684210526315
# StackingClassifier 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!