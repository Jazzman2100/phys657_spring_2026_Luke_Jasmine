import numpy as np
import matplotlib.pyplot as plt
import math
# define pure Gaussian noise
sigma = 0.3
size = 1000

x = np.linspace(0, 1, size)

y1 = np.sin(2*math.pi*x)

y2 = np.random.normal(loc=0, scale=sigma, size=size)

z = np.polyfit(x, y1+y2, 2)
# y3 = np.random.normal(x, mu, sigma)

plt.figure(figsize=(8, 5))
plt.scatter(x, y1+y2, color='blue')
plt.plot(x, z, color='red')
plt.title('Test plot')
plt.xlabel('value')
plt.ylabel('Distribution')
plt.legend()
plt.grid(True)
plt.show()