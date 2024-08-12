import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#1. data

x1_datasets = np.array([range(100), range(301,401)]).T
                        #삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
                        #원유, 환율, 금시세                    
x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T


y1 = np.array(range(3001, 3101)) #한강의 화씨 온도
y2 = np.array(range(13001,13101)) #비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test =train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, train_size=0.9, random_state=5656
)

