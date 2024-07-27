import numpy as np
from sklearn.preprocessing import StandardScaler

#1. data
data = np.array(([1,2,3,1],
                 [4,5,6,2],
                 [7,8,9,3],
                 [10,11,12,114],
                 [13,14,15,115]))


#1. 평균
means = np.mean(data, axis=0)
print('average:', means)  # average: [ 7.  8.  9. 47.]


#2. 모집단 분산 (n빵) #  ddof=0 디폴트
population_variances = np.var(data, axis=0)
print("variances:", population_variances) # variances: [  18.   18.   18. 3038.]


#3. 표본 분산(n-1빵)
variances = np.var(data, axis=0, ddof=1) #ddof가 모얌
print("표본 분산 :" , variances) #표본 분산 : [  22.5   22.5   22.5 3797.5]

#4. 표준 편차
std = np.std(data, axis=0, ddof=1)
print("표준편차 : ", std) #표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5. StandardScaler #모집단분산을 이용하여 해당 표준편차로 나눔. 모표준편차 z=x-평균/표준편차
scaler=StandardScaler()
scaled_data = scaler.fit_transform(data)

print("standard : \n", scaled_data) 
#[[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]



