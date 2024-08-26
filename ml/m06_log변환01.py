import numpy as np
import matplotlib.pyplot as plt

data=np.random.exponential(scale=2.0, size=1000)
print(data)
print(data.shape)
print(np.min(data), np.max(data)) #6.508678730990796e-05, 11.548598331059617

log_data = np.log(data)
 
# 원본 데이터
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='yellow', alpha=0.5)
plt.title('Original')
# plt.show()


#로그변환 데이터 히스토그램 그리자
plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='purple', alpha=0.5)
plt.title('Log Transformed')
plt.show()