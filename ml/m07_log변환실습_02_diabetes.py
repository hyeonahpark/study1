from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#1. data
dataset = load_diabetes()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# print(df)
# df.boxplot()
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

#[실습]
#R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=52151)

##################### y 로그 변환####################################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
##################### y 로그 변환####################################

#2. model 랜덤 포레스트 모델 학습
model = RandomForestRegressor(random_state= 52151, max_depth=5, min_samples_split=3)

#3. fit
model.fit(x_train, y_train)

#4. predict
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print("score : ", score) 

#변환 x  score :  0.5187753598635303
#변환    score :  0.5366610211847753