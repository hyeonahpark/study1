"""
결측치처리
1. 삭제 - 행 또는 열
2. 임의의 값
  평균 : mean
  중위 : median
  0 : fillna
  앞값 : ffill
  뒷값 : bfill
  특정값 : 777(먼가 조건을 같이 넣기)
  기타등등
3.interpolate
4. 모델 : .predict
5. 부스팅 계열 모델 : 이상치, 결측치 처리에 자유로움
"""


import pandas as pd
import numpy as np

data = pd.DataFrame(([2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]))
print(data)

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

# 0. 결측치 확인
print(data.isnull()) #bool 형태로 나옴
print(data.isnull().sum()) #결측치 갯수
print(data.info())

# 1. 결측치 삭제
# print(data.dropna()) # true 있는 행 다 삭제
# print(data.dropna(axis=0)) # 행 삭제
# print(data.dropna(axis=1)) # 열 삭제


#2-1. 특정값 - 평균
means = data.mean()
print(means)
data2=data.fillna(means) #열 기준 평균값으로
print(data2)

#2-2. 특정값 - 중위값
med = data.median()
print(med)
data3=data.fillna(med) #열 기준 평균값으로
print(data3)

#2-3. 특정값 - 0채우기 / 임의의값 채우기
data4 = data.fillna(0)
print(data4)

data4_2 = data.fillna(7777)
print(data4_2)

#2-4. 특정값 - ffill (통상 마지막값에,)
data5 = data.ffill()
#data5 = data.fillna(method= 'ffill')
print(data5)

#2-5. 특정값 - bfill (통상 마지막값에,)
data6 = data.bfill()
#data5 = data.fillna(method= 'bfill')
print(data6)


################################## 특정 컬럼만 ################################
means = data['x1'].mean()
print(means) # 6.5

meds = data['x4'].median()
print(meds) # 6.0

data['x1'] = data['x1'].fillna(means)
data['x4'] = data['x4'].fillna(meds)
data['x2'] = data['x2'].ffill()

print(data)
