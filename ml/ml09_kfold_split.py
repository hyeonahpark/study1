import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.svm import SVC
import pandas as pd

#1. data
datasets = load_iris() 

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

print(df)
# # dtc = DecisionTreeClassifier(random_state=156)

n_splits=3
kfold = KFold(n_splits=n_splits, shuffle=False, 
              #random_state=333
              )


for train_index, val_index in kfold.split(df) : 
    print("=============================================")
    print(train_index, '\n', val_index)
    print('훈련데이터 개수 : ', len(train_index), " ", 
          "검증데이터 갯수", len(val_index))
    
    