import numpy as np
import matplotlib.pyplot as plt

#Making the N variable to be assigned. Make it 10, 100, 1000
Number = 10

#Making the noise for a gaussian range for ten data points
noise = np.random.normal(0,0.3,Number)

#Making the random x values. Training data
x_values = np.random.uniform(0,1,Number)

#Making the sine function y values
y_values = np.sin(2 * np.pi * x_values)

#Making the Polynomial fits from 1 through 9
poly1 = np.polyfit(x_values,y_values + noise,1)
f1 = np.poly1d(poly1)
poly2 = np.polyfit(x_values,y_values + noise,2)
f2 = np.poly1d(poly2)
poly3 = np.polyfit(x_values,y_values + noise,3)
f3 = np.poly1d(poly3)
poly4 = np.polyfit(x_values,y_values + noise,4)
f4 = np.poly1d(poly4)
poly5 = np.polyfit(x_values,y_values + noise,5)
f5 = np.poly1d(poly5)
poly6 = np.polyfit(x_values,y_values + noise,6)
f6 = np.poly1d(poly6)
poly7 = np.polyfit(x_values,y_values + noise,7)
f7 = np.poly1d(poly7)
poly8 = np.polyfit(x_values,y_values + noise,8)
f8 = np.poly1d(poly8)
poly9 = np.polyfit(x_values,y_values + noise,9)
f9 = np.poly1d(poly9)

#Making graph x values
x_graph_values = np.linspace(0,1,100)

fig, axs = plt.subplots()
plt.title('Activity 1 part b. Polynomial fit')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.plot(x_values, y_values + noise, c='k')
plt.plot(x_graph_values, f1(x_graph_values), c='b')
plt.plot(x_graph_values, f2(x_graph_values), c='r')
plt.plot(x_graph_values, f3(x_graph_values), c='g')

plt.plot(x_graph_values, f4(x_graph_values), c='c')
plt.plot(x_graph_values, f5(x_graph_values), c='navy')
plt.plot(x_graph_values, f6(x_graph_values), c='gold')
plt.plot(x_graph_values, f7(x_graph_values), c='m')
plt.plot(x_graph_values, f8(x_graph_values), c='orange')
plt.plot(x_graph_values, f9(x_graph_values), c='olive')
plt.legend(['data', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th'])


plt.grid(True)
plt.show()
