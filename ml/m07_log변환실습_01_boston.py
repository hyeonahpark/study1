import numpy as np
from tensorflow.keras.models import Sequential, load_model #load_model : model 을 불러옴
from tensorflow.keras.layers import Dense
import sklearn as sk
print(sk.__version__) #0.24.2
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time
from keras.layers import Dropout
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#1.data
dataset=load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

x=dataset.data
y=dataset.target

df['target'] = dataset.target

x = df.drop(['target'], axis=1).copy()
y = df['target']



x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, random_state=1186)

############################X 데이터 로그변환##########################################
# x['TAX'] = np.log1p(x['TAX'])
# x['B'] = np.log1p(x['B'])
############################X 데이터 로그변환##########################################

##################### y 로그 변환####################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##################### y 로그 변환####################################

#2. model 랜덤 포레스트 모델 학습
model = RandomForestRegressor(random_state= 1186, max_depth=5, min_samples_split=3)

#3. fit
model.fit(x_train, y_train)

#4. predict
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print("score : ", score) 


# 둘다 변환X : score :  0.8733382693916973
# X만변환 :  score :  0.8733382693916973
# Y만변환 : score :  0.8582224059265878
#둘다 변환  0.8582224059265878
