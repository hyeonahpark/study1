# 다른 말로 swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def mish(x):
#     return x * np.tanh(np.log(1+np.exp(x)))

mish= lambda x : x * np.tanh(np.log(1+np.exp(x)))

# x 곱하기 sigmoid 
# 문제점 : Relu보다 계산량이 많아서 모델이 커질수록 부담스러움

y = mish(x)

plt.plot(x,y)
plt.grid()
plt.show()

#7. elu
#8. selu
#9. leaky_relu