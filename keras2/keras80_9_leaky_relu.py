import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5, 5, 0.1)

def leaky_relu(x) :
    return (x<=0)*0.01*x + (x>0) * (np.maximum(0, x))

#leaky_relu = lambda x : (x>0)*0.01*x + (x<=0) * (np.maximum(0, x))

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()