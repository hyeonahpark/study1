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

dates = ['10/11/2024', '10/12/2024','10/13/2024',
         '10/14/2024', '10/15/2024', '10/16/2024']
dates = pd.to_datetime(dates)
print(dates)


print("=================================")
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index = dates)
print(ts)
# 2024-10-11     2.0
# 2024-10-12     NaN
# 2024-10-13     NaN
# 2024-10-14     8.0
# 2024-10-15    10.0
# 2024-10-16     NaN