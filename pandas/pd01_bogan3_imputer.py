import pandas as pd
import numpy as np

data = pd.DataFrame(([2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]))
#print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

imputer = SimpleImputer()
data2 = imputer.fit_transform(data) #default : 평균
print(data2)

imputer = SimpleImputer(strategy='mean') #평균
data3 = imputer.fit_transform(data)
print(data3)

imputer = SimpleImputer(strategy='median') #중위값
data4 = imputer.fit_transform(data)
print(data4)

imputer = SimpleImputer(strategy='most_frequent') #최빈값
data5 = imputer.fit_transform(data)
print(data5)

imputer = SimpleImputer(strategy='constant', fill_value=7777) # 상수, 특정값
data6 = imputer.fit_transform(data)
print(data6)

imputer = KNNImputer() #KNN 알고리즘으로 결측치 처리
data7 = imputer.fit_transform(data)
print(data7)

imputer = IterativeImputer() #// MICE 방식
data8 = imputer.fit_transform(data)
print(data8)
print(np.__version__) #1.26.3

np.float = float

#pip install impyute
from impyute.imputation.cs import mice
data9 = mice(data.values,
             n=10,
             seed=777)
print(data9)