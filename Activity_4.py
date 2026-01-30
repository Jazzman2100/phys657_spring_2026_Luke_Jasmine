import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

'''
This python File is fairly straight forward. I plotted the Bayesian Curve Fitting graph here and nothing else. 
One note about this file and how I did this was how I did use Testing and Training data (both N=100), mainly out of habit.
I wasn't entirely sure if I was suppose to do this for this type of method or not though. 
The result for either of them like this look pretty similiar though. 
'''

#Making the N variable to be assigned. Make it 10, 100, 1000
Number = 100
Graph_number = 100

#Making the noise for a gaussian range for ten data points
noise = np.random.normal(0,0.3,Number)

#Making the random x values. Training data
x_values = np.random.uniform(0,1,Number)

#Making the sine function y values. Also training data
y_values = np.sin(2 * np.pi * x_values)
y_data = y_values + noise

#Making graph x values. Testing data. Will be used to calcuate the error
x_graph_values = np.linspace(0,1, Graph_number)

y_graph_values = np.sin(2*np.pi* x_graph_values)
testing_noise = np.random.normal(0,0.3, Graph_number)
y_graph_data = y_graph_values + testing_noise

alpha = 5E-3
beta = 11.1

def Polynomial(data,power):
    answer = data**power
    return answer

number_of_powers = np.arange(10)

#For training
phi_train = []
for i in number_of_powers:
    result1 = Polynomial(x_values, i)
    phi_train.append(result1)
#Will make a 2D Array (nest array) first selection is what power you put the data in, next is the data itself
phi_train_ak = np.array(phi_train)

#For testing
phi_test = []
for i in number_of_powers:
    result2 = Polynomial(x_graph_values, i)
    phi_test.append(result2)
#Will make a 2D Array (nest array) first selection is what power you put the data in, next is the data itself
phi_test_ak = np.array(phi_test)

S_inv = alpha * np.eye(len(number_of_powers)) + beta * (phi_train @ np.transpose(phi_train))
S_matrix = np.linalg.inv(S_inv)
mean = ak.flatten(beta * (np.transpose(phi_test)) @ S_matrix @ (phi_test @ np.vstack(y_graph_data)))
variance = np.diag((1/beta) + np.transpose(phi_test) @ S_matrix @ phi_test)

fig, axs = plt.subplots()
plt.title(f'Activity 4. Bayesian curve Fitting')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.scatter(x_graph_values, y_graph_data, c='b')
plt.plot(x_graph_values, mean, c='r')
plt.fill_between(x_graph_values, mean - np.sqrt(variance), mean + np.sqrt(variance), alpha=0.5 )

plt.grid(True)
plt.show()
plt.close()

