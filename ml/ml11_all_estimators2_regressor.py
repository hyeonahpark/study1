import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston
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
x, y = load_boston(return_X_y=True) # 바로x와 y로 들어감 !

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.1, random_state=1186)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

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


# 모델의 갯수 :  54
# ARDRegression 의 정답률 :  0.6997594278589961
# AdaBoostRegressor 의 정답률 :  0.8349947691658743
# BaggingRegressor 의 정답률 :  0.8786748661550801
# BayesianRidge 의 정답률 :  0.7081081175956494   
# CCA 의 정답률 :  0.583715635433762
# DecisionTreeRegressor 의 정답률 :  0.6753850540649018
# DummyRegressor 의 정답률 :  -1.4523875347727255e-05
# ElasticNet 의 정답률 :  0.7226527234958685
# ElasticNetCV 의 정답률 :  0.7062396887055422
# ExtraTreeRegressor 의 정답률 :  0.8119989924726947
# ExtraTreesRegressor 의 정답률 :  0.9059219410559155
# GammaRegressor 의 정답률 :  0.7617263445172183
# GaussianProcessRegressor 의 정답률 :  0.6172478922462095
# GradientBoostingRegressor 의 정답률 :  0.8845018239184455
# HistGradientBoostingRegressor 의 정답률 :  0.8878407547172442
# HuberRegressor 의 정답률 :  0.7302761975719668
# IsotonicRegression 은 바보 멍충이!!!
# KNeighborsRegressor 의 정답률 :  0.7551049270925436
# KernelRidge 의 정답률 :  -7.266579704438318
# Lars 의 정답률 :  0.6991154708488959
# LarsCV 의 정답률 :  0.7117064396429021
# Lasso 의 정답률 :  0.6892463329745535
# LassoCV 의 정답률 :  0.7010587186534332
# LassoLars 의 정답률 :  -1.4523875347727255e-05
# LassoLarsCV 의 정답률 :  0.6991154708488959
# LassoLarsIC 의 정답률 :  0.7055227980436494
# LinearRegression 의 정답률 :  0.6991154708488958
# LinearSVR 의 정답률 :  0.7225865358064121
# MLPRegressor 의 정답률 :  0.8126538044583721
# MultiOutputRegressor 은 바보 멍충이!!!
# MultiTaskElasticNet 은 바보 멍충이!!!
# MultiTaskElasticNetCV 은 바보 멍충이!!!
# MultiTaskLasso 은 바보 멍충이!!!
# MultiTaskLassoCV 은 바보 멍충이!!!
# NuSVR 의 정답률 :  0.6721297302827566
# OrthogonalMatchingPursuit 의 정답률 :  0.5073646876966904
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6668413305446459
# PLSCanonical 의 정답률 :  -3.292878953314677
# PLSRegression 의 정답률 :  0.7387093078690532
# PassiveAggressiveRegressor 의 정답률 :  0.41265210830576526
# PoissonRegressor 의 정답률 :  0.7963073234152259
# RANSACRegressor 의 정답률 :  0.5469816157681961
# RadiusNeighborsRegressor 은 바보 멍충이!!!
# RandomForestRegressor 의 정답률 :  0.8863241641372599
# RegressorChain 은 바보 멍충이!!!
# Ridge 의 정답률 :  0.7009888292239703
# RidgeCV 의 정답률 :  0.7009888292237552
# SGDRegressor 의 정답률 :  0.7089183919464608
# SVR 의 정답률 :  0.6689936085513024
# StackingRegressor 은 바보 멍충이!!!
# TheilSenRegressor 의 정답률 :  0.5520368400347379
# TransformedTargetRegressor 의 정답률 :  0.6991154708488958
# TweedieRegressor 의 정답률 :  0.7225915269177892
# VotingRegressor 은 바보 멍충이!!!