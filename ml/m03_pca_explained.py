#train_test_split 후 스케일링 후 PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np

#1. data
datasets = load_iris()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150, )

x_train, x_test, y_train, y_test = train_test_split(
    
    x, y, train_size=0.9, random_state = 1186, shuffle=True, stratify=y,
) #stratify=y : y의 라벨에 맞춰서 train_test_split의 비율을 정함.

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

for i in range(x.shape[1], 0, -1) :
    pca = PCA(n_components = i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)  
    model = RandomForestClassifier(random_state=1186)
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print('n_components: ', i)
    print('model.score: ', results)


# pca = PCA(n_components = 2)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)  
    
# print(x_train.shape)
# print(x_test.shape)


# #2. modeling
# model = RandomForestClassifier(random_state=1186)

# #3. compile
# model.fit(x_train, y_train)

# #4. predict
# results = model.score(x_test, y_test)
# # print(x.shape)
# print('model.score: ', results) # model.score:  1.0 -> accuracy
#RandomForestRegressor는 score 뽑으면 R2 score 뽑아줌

# (135, 4)
# (15, 4)
# model.score:  1.0

# (135, 3)
# (15, 3)
# model.score:  1.0

# (135, 2)
# (15, 2)
# model.score:  0.8666666666666667

# (135, 1)
# (15, 1)
# model.score:  0.9333333333333333

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()