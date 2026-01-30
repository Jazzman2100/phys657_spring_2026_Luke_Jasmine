import numpy as np
import matplotlib.pyplot as plt

#This python file is just used to make the data and plot it out at different N values.

#Making the N variable to be assigned. Make it 10, 100, 1000
Number = 100

#Making the noise for a gaussian range for ten data points
noise = np.random.normal(0,0.3,Number)

#Making the random x values
x_values = np.random.uniform(0,1,Number)

#Making the sine function y values
y_values = np.sin(2 * np.pi * x_values)

fig, axs = plt.subplots()
plt.title('Activity 1 part a. Sine function plus random noise.')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.scatter(x_values, y_values + noise)
plt.grid(True)
plt.show()
