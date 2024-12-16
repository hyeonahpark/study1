import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
plt.rcParams['font.family'] = 'Malgun Gothic'

#1. data
np.random.seed(777)
x = 2*np.random.rand(100, 1) -1 # -1부터 1까지 난수생성
#print(x)
y = 3*x**2 + 2*x + 1 + np.random.randn(100, 1) # y = 3x^2 + 2x + 1 + noise

# rand : 0 ~ 1 사이의 균일분포 난수
# randn : 평균이 0, 표준 편차가 1 정규 분포를 따르는 난수

pf = PolynomialFeatures(degree = 2, include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly) # x의 제곱 값을 가지는 열이 하나 더 생성됨

#2. model
model = LinearRegression()
model2 = LinearRegression()

#3. training
model.fit(x, y)
model2.fit(x_poly, y)

#원래 데이터 그리자.
plt.scatter(x,y,color='blue', label='Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')
#plt.show()

#다항식 회귀 그래프 그리기
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)
plt.plot(x_plot, y_plot, color = 'red', label = 'Polynomial Regression')
plt.plot(x_plot, y_plot2, color = 'green', label = '기냥')
plt.legend()
plt.show()