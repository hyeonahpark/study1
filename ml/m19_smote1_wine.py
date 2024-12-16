import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(33)

#1. data
datasets = load_wine()
x=datasets.data
y=datasets['target']

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# 2 40개 삭제하기 -> 불균형 데이터
x = x[:-39]
y = y[:-39]
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71,  8], dtype=int64))
# y=pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123, stratify=y)
#train만 증폭, test 증폭도 해도 되지만 성능보장 x 과적합 우려

"""
#2. model
model = Sequential()
model.add(Dense(10, input_shape=(13, )))
model.add(Dense(3, activation='softmax'))

#3. compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. predict
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


#f1 score
# y_predict = np.round(model.predict(x_test))
y_predict = model.predict(x_test)
y_predict= np.argmax(y_predict, axis=1)
# y_test = np.argmax(np.array(y_test), axis=1)
print(y_predict)


acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print('acc : ', acc)
print('f1 : ', f1)

# acc :  0.9142857142857143
# f1 :  0.9384902143522833


#2. model
model = XGBClassifier()

#3. compile
model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)], #validation
          verbose=True,
          )


#4. predict
results = model.score(x_test, y_test)
print('model.score : ', results)

#지표 f1 score
y_predict = model.predict(x_test)
print('f1_score : ', f1_score(y_test, y_predict, average='macro'))
#model.score :  1.0
# f1_score :  1.0
"""

########################################### smote 적용  #####################################################
# pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('sklearn : ', sk.__version__) #sklearn :  1.5.1

print('증폭 전 :', np.unique(y_train, return_counts=True)) #증폭 전 : (array([0, 1, 2]), array([44, 53,  6], dtype=int64))

smote = SMOTE(random_state = 7777)
x_train, y_train = smote.fit_resample(x_train, y_train)
print('증폭 후 :', np.unique(y_train, return_counts=True)) #증폭 후 : (array([0, 1, 2]), array([53, 53, 53], dtype=int64))
print(pd.value_counts(y_train))
# 0    53
# 1    53
# 2    53

#2. model
model = Sequential()
model.add(Dense(10, input_shape=(13, )))
model.add(Dense(3, activation='softmax'))

#3. compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. predict
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


#f1 score
# y_predict = np.round(model.predict(x_test))
y_predict = model.predict(x_test)
y_predict= np.argmax(y_predict, axis=1)
# y_test = np.argmax(np.array(y_test), axis=1)
print(y_predict)


acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print('acc : ', acc)
print('f1 : ', f1)
########################################### smote 적용 끝  #####################################################

###smote 적용 전 
# acc :  0.8571428571428571
# f1 :  0.5970961887477314

###smote 적용 후
# acc :  0.8857142857142857
# f1 :  0.6259259259259259