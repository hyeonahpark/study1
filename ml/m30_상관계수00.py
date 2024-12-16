from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd 

#1. data
datasets= load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)

df['Target'] = y
print(df)


print("==============================상관계수 히트맵 =============================")
print(df.corr())


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__) #0.11.2
print(matplotlib.__version__) #3.4.3
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True, #표 안에 수치 명시
            cbar=True) #사이드바
plt.show()