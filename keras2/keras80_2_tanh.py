import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
# y = np.tanh(x)

# p_exp_x = np.exp(x)
# m_exp_x = np.exp(-x)
# y= (p_exp_x - m_exp_x)/(p_exp_x  + m_exp_x)

tanh = lambda x : (np.exp(x) - np.exp(-x))/(np.exp(x)  + np.exp(-x))

y = tanh(x)

plt.plot(x,y)
plt.grid()
plt.show()