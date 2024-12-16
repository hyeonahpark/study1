#iris
#cancer
#wine
#digits

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score, cross_validate, KFold, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import time
import warnings
warnings.filterwarnings('ignore')


#1. data

iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)


datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)

#2. modeling
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1186)

start_time = time.time()

for index, value in enumerate(datasets) : #enumerate는 인덱스 값도 같이 반환해줌
    x, y =value
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1186)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print('\n=========================', data_name[index], '=============================')
    
    my_max = 0
    # model_name=[]
    # model_acc=[]
    
    for name, model in all:
        try :
            #2. model
            model = model()
            #3. training
            model.fit(x_train, y_train)
            
            #3. training
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            # print('=============', name, '================')
            acc = model.score(x_test, y_test)
            # print("ACC : ", scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
            aver_acc = round(np.mean(scores), 4)
            
            # model_name.append(name)
            # model_acc.append(aver_acc)
            
        except:
            continue
            # print(name, '은 바보 멍충이!!!')
    
        if my_max < aver_acc:
            my_max = aver_acc 
            model_name = name   
        print("최고모델 : ", model_name , "최대 acc :", my_max)       
    # model_name=np.array(model_name)
    # model_acc=np.array(model_acc)
    # max_index=np.where(model_acc==np.max(model_acc))
    # print('최고모델 :', model_name[max_index], 'ACC :', model_acc[max_index])          

end_time = time.time()
print('걸린시간 : ', round(end_time-start_time,2), '초')


# ========================= 아이리스 =============================
# 최고모델 :  LinearDiscriminantAnalysis 최대 acc : 0.9822
# ========================= 캔서 =============================
# 최고모델 :  LogisticRegressionCV 최대 acc : 0.9812
# ========================= 와인 =============================
# 최고모델 :  CalibratedClassifierCV 최대 acc : 0.9926
# ========================= 디지트 =============================
# 최고모델 :  ExtraTreesClassifier 최대 acc : 0.9807

