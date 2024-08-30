import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston
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
x, y = load_boston(return_X_y=True) # 바로x와 y로 들어감 !

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

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
        n_splits=5
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

        #3. training
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        acc = model.score(x_test, y_test)
        y_predcit = cross_val_predict(model, x_test, y_test, cv=kfold)

        print('=============', name, '================')
        print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
        acc = accuracy_score(y_test, y_predcit)
        print('cross_val_predict ACC : ', acc)
        #4. predict
        print(name, '의 정답률 : ', acc)
        
    except:
        print(name, '은 바보 멍충이!!!')
        
        
# 모델의 갯수 :  54
# ============= ARDRegression ================
# ACC :  [0.7637843  0.72388757 0.69957738 0.8160395  0.7203927 ] 
#  평균 ACC :  0.7447
# ARDRegression 은 바보 멍충이!!!
# ============= AdaBoostRegressor ================
# ACC :  [0.77660287 0.67790819 0.77669107 0.830886   0.75222378] 
#  평균 ACC :  0.7629
# AdaBoostRegressor 은 바보 멍충이!!!
# ============= BaggingRegressor ================
# ACC :  [0.81696149 0.67580056 0.77361177 0.903677   0.80923147] 
#  평균 ACC :  0.7959
# BaggingRegressor 은 바보 멍충이!!!
# ============= BayesianRidge ================
# ACC :  [0.76833673 0.73832705 0.70144118 0.82366408 0.72196689]
#  평균 ACC :  0.7507
# BayesianRidge 은 바보 멍충이!!!
# ============= CCA ================
# ACC :  [0.75029309 0.55403084 0.57227366 0.72552266 0.71025537]
#  평균 ACC :  0.6625
# CCA 은 바보 멍충이!!!
# ============= DecisionTreeRegressor ================
# ACC :  [0.75434623 0.5224507  0.73787981 0.77121553 0.68711268]
#  평균 ACC :  0.6946
# DecisionTreeRegressor 은 바보 멍충이!!!
# ============= DummyRegressor ================
# ACC :  [-1.72624561e-06 -2.71417968e-02 -2.08864926e-03 -7.29161810e-04
#  -3.71766465e-02]
#  평균 ACC :  -0.0134
# DummyRegressor 은 바보 멍충이!!!
# ============= ElasticNet ================
# ACC :  [0.67498833 0.69019061 0.67058104 0.73887371 0.59284086]
#  평균 ACC :  0.6735
# ElasticNet 은 바보 멍충이!!!
# ============= ElasticNetCV ================
# ACC :  [0.76752501 0.73815673 0.70124638 0.82358179 0.71735711] 
#  평균 ACC :  0.7496
# ElasticNetCV 은 바보 멍충이!!!
# ============= ExtraTreeRegressor ================
# ACC :  [0.6710463  0.48570819 0.54849102 0.53104851 0.71724046]
#  평균 ACC :  0.5907
# ExtraTreeRegressor 은 바보 멍충이!!!
# ============= ExtraTreesRegressor ================
# ACC :  [0.86183976 0.78840957 0.86251745 0.89479415 0.89700157] 
#  평균 ACC :  0.8609
# ExtraTreesRegressor 은 바보 멍충이!!!
# ============= GammaRegressor ================
# ACC :  [0.6890626  0.71327889 0.69856452 0.73946445 0.60205046]
#  평균 ACC :  0.6885
# GammaRegressor 은 바보 멍충이!!!
# ============= GaussianProcessRegressor ================
# ACC :  [ 0.46086167 -0.11225907 -0.11984659  0.56011678  0.39734284] 
#  평균 ACC :  0.2372
# GaussianProcessRegressor 은 바보 멍충이!!!
# ============= GradientBoostingRegressor ================
# ACC :  [0.89050178 0.70804138 0.86384702 0.90220084 0.80839531] 
#  평균 ACC :  0.8346
# GradientBoostingRegressor 은 바보 멍충이!!!
# ============= HistGradientBoostingRegressor ================
# ACC :  [0.84519644 0.74460725 0.81128576 0.8994018  0.78813047] 
#  평균 ACC :  0.8177
# HistGradientBoostingRegressor 은 바보 멍충이!!!
# ============= HuberRegressor ================
# ACC :  [0.75392576 0.74404818 0.67817094 0.85219789 0.68524288] 
#  평균 ACC :  0.7427
# HuberRegressor 은 바보 멍충이!!!
# IsotonicRegression 은 바보 멍충이!!!
# ============= KNeighborsRegressor ================
# ACC :  [0.79460796 0.75848852 0.72302138 0.77943721 0.73443496]
#  평균 ACC :  0.758
# KNeighborsRegressor 은 바보 멍충이!!!
# ============= KernelRidge ================
# ACC :  [-4.6077503  -7.05651733 -6.60010683 -6.94611548 -4.68020007]
#  평균 ACC :  -5.9781
# KernelRidge 은 바보 멍충이!!!
# ============= Lars ================
# ACC :  [0.76573478 0.73742552 0.70207279 0.81909794 0.72584181]
#  평균 ACC :  0.75
# Lars 은 바보 멍충이!!!
# ============= LarsCV ================
# ACC :  [0.76965734 0.7368174  0.70122757 0.82234977 0.72088172] 
#  평균 ACC :  0.7502
# LarsCV 은 바보 멍충이!!!
# ============= Lasso ================
# ACC :  [0.71596818 0.68472203 0.69246547 0.7557017  0.62778283]
#  평균 ACC :  0.6953
# Lasso 은 바보 멍충이!!!
# ============= LassoCV ================
# ACC :  [0.76632767 0.73748514 0.70131922 0.82166485 0.71939299] 
#  평균 ACC :  0.7492
# LassoCV 은 바보 멍충이!!!
# ============= LassoLars ================
# ACC :  [-1.72624561e-06 -2.71417968e-02 -2.08864926e-03 -7.29161810e-04
#  -3.71766465e-02]
#  평균 ACC :  -0.0134
# LassoLars 은 바보 멍충이!!!
# ============= LassoLarsCV ================
# ACC :  [0.76585444 0.73759653 0.70082783 0.82231052 0.72071395] 
#  평균 ACC :  0.7495
# LassoLarsCV 은 바보 멍충이!!!
# ============= LassoLarsIC ================
# ACC :  [0.76665868 0.73747194 0.69010174 0.79996186 0.64278148]
#  평균 ACC :  0.7274
# LassoLarsIC 은 바보 멍충이!!!
# ============= LinearRegression ================
# ACC :  [0.76573478 0.73742552 0.70207279 0.81909794 0.72584181]
#  평균 ACC :  0.75
# LinearRegression 은 바보 멍충이!!!
# ============= LinearSVR ================
# ACC :  [0.73604754 0.73847569 0.66837799 0.84992481 0.6673477 ]
#  평균 ACC :  0.732
# LinearSVR 은 바보 멍충이!!!
# ============= MLPRegressor ================
# ACC :  [0.65105133 0.73138356 0.57522151 0.65560503 0.53931272] 
#  평균 ACC :  0.6305
# MLPRegressor 은 바보 멍충이!!!
# MultiOutputRegressor 은 바보 멍충이!!!
# MultiTaskElasticNet 은 바보 멍충이!!!
# MultiTaskElasticNetCV 은 바보 멍충이!!!
# MultiTaskLasso 은 바보 멍충이!!!
# MultiTaskLassoCV 은 바보 멍충이!!!
# ============= NuSVR ================
# ACC :  [0.62187248 0.60856071 0.62150987 0.72543448 0.55885038]
#  평균 ACC :  0.6272
# NuSVR 은 바보 멍충이!!!
# ============= OrthogonalMatchingPursuit ================
# ACC :  [0.58335741 0.34623786 0.45991677 0.54585853 0.43082104]
#  평균 ACC :  0.4732
# OrthogonalMatchingPursuit 은 바보 멍충이!!!
# ============= OrthogonalMatchingPursuitCV ================
# ACC :  [0.75785884 0.68182217 0.65994026 0.75756237 0.66419401]
#  평균 ACC :  0.7043
# OrthogonalMatchingPursuitCV 은 바보 멍충이!!!
# ============= PLSCanonical ================
# ACC :  [-1.66465159 -3.27647028 -1.76924063 -2.97285785 -1.43760136]
#  평균 ACC :  -2.2242
# PLSCanonical 은 바보 멍충이!!!
# ============= PLSRegression ================
# ACC :  [0.77566098 0.70895426 0.65827522 0.77986243 0.70166981]
#  평균 ACC :  0.7249
# PLSRegression 은 바보 멍충이!!!
# ============= PassiveAggressiveRegressor ================
# ACC :  [ 0.57917333 -0.02056627  0.62251602  0.76886629  0.58069982]
#  평균 ACC :  0.5061
# PassiveAggressiveRegressor 은 바보 멍충이!!!
# ============= PoissonRegressor ================
# ACC :  [0.82045295 0.77014702 0.76169697 0.86728348 0.77095467]
#  평균 ACC :  0.7981
# PoissonRegressor 은 바보 멍충이!!!
# ============= RANSACRegressor ================
# ACC :  [0.47751231 0.58894566 0.5358193  0.8042472  0.57719811] 
#  평균 ACC :  0.5967
# RANSACRegressor 은 바보 멍충이!!!
# RadiusNeighborsRegressor 은 바보 멍충이!!!
# ============= RandomForestRegressor ================
# ACC :  [0.84282237 0.74875945 0.83106787 0.90877634 0.80668241] 
#  평균 ACC :  0.8276
# RandomForestRegressor 은 바보 멍충이!!!
# RegressorChain 은 바보 멍충이!!!
# ============= Ridge ================
# ACC :  [0.76650109 0.7376058  0.70192238 0.82028678 0.72517068]
#  평균 ACC :  0.7503
# Ridge 은 바보 멍충이!!!
# ============= RidgeCV ================
# ACC :  [0.76650109 0.73883476 0.7011717  0.82028678 0.71882099]
#  평균 ACC :  0.7491
# RidgeCV 은 바보 멍충이!!!
# ============= SGDRegressor ================
# ACC :  [0.7693485  0.73154369 0.69621496 0.82272988 0.72356929]
#  평균 ACC :  0.7487
# SGDRegressor 은 바보 멍충이!!!
# ============= SVR ================
# ACC :  [0.65087294 0.60832403 0.64120604 0.743567   0.58015819]
#  평균 ACC :  0.6448
# SVR 은 바보 멍충이!!!
# StackingRegressor 은 바보 멍충이!!!
# ============= TheilSenRegressor ================
# ACC :  [0.63199044 0.70578864 0.51587393 0.70189616 0.50312782] 
#  평균 ACC :  0.6117
# TheilSenRegressor 은 바보 멍충이!!!
# ============= TransformedTargetRegressor ================       
# ACC :  [0.76573478 0.73742552 0.70207279 0.81909794 0.72584181] 
#  평균 ACC :  0.75
# TransformedTargetRegressor 은 바보 멍충이!!!
# ============= TweedieRegressor ================
# ACC :  [0.65412413 0.69506543 0.65367329 0.73630725 0.58228812] 
#  평균 ACC :  0.6643
# TweedieRegressor 은 바보 멍충이!!!
# VotingRegressor 은 바보 멍충이!!!