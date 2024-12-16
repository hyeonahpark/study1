import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)
print(x)
#[[ 0  1  2]
# [ 3  4  5]
# [ 6  7  8]
# [ 9 10 11]]

pf = PolynomialFeatures(degree = 2, include_bias=False) # true 가 default
x_pf = pf.fit_transform(x)
print(x_pf)
#[[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
# [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
# [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
# [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]

pf2 = PolynomialFeatures(degree =3, include_bias=False)
x_pf2 = pf2.fit_transform(x)
print(x_pf2)
print(x_pf2.shape) #(4, 19)