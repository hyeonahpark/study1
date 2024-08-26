from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

(x_train, _), (x_test, _) = mnist.load_data()

# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis = 0)
# print(x.shape)

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])

########################[실습] ########################
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95이상
# 0.99이상
# 0.999이상
# 1.0일 때 몇개??

# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)

pca = PCA(n_components = 28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
print(sum(evr))
evr_cumsum = np.cumsum(evr)
print("0.95이상 : ",np.argmax(evr_cumsum>=0.95)+1)
print("0.99이상 : ",np.argmax(evr_cumsum>=0.99)+1)
print("0.999이상 : ",np.argmax(evr_cumsum>=0.999)+1)
print("1.0 : ", np.argmax(evr_cumsum>=1.0)+1)

# 0.95이상 :  154
# 0.99이상 :  331
# 0.999이상 :  486
# 1.0 :  713

"""
for i in range(3, 0, -1) :
    pca = PCA(n_components = i)
    x = pca.fit_transform(x)
    model = RandomForestClassifier(random_state=1186)
    model.fit(x_train)
    results = model.score(x_test)
    print('n_components: ', i)
    print('model.score: ', results)
    evr = pca.explained_variance_ratio_ # 설명가능한 변화율
    evr_cumsum = np.cumsum(evr)
    
    for i in range(3, 0, -1) :
        if evr_cumsum[i] >=0.95 :
            x1 = x1+1
        if evr_cumsum[i] >=0.99 :
            x2 = x2+1
        if evr_cumsum[i] >=0.999 :
            x3 = x3+1
        if evr_cumsum[i] == 1.0 :
            x4 = x4+1
    print(x1, x2, x3,x4)
        

    
# evr_cumsum = np.cumsum(evr)
# print(evr_cumsum)
# import matplotlib.pyplot as plt
# plt.plot(evr_cumsum)
# plt.grid()
# plt.show()
"""