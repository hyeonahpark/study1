import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def selu(x, a=2):
    return (x>0)*x + (x<=0) * (a * (np.exp(x) -1))

a = 2 
#elu = lambda x : (x>0)*x + (x<=0) * (a * (np.exp(x) -1))
#
y = selu(x)

plt.plot(x,y)
plt.grid()
plt.show()

#7. elu
#8. selu
#9. leaky_relu