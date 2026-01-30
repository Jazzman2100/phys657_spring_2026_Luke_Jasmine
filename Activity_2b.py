import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures


'''
This python file was used to plot the E_rms vs the lambda values. This graph was really tricky as the random data and the fitting didn't mean that I would get
a consistent result that matched the figure. But I got the one I showed in the analysis through pure luck. Most of the time it just looks like the Testing E_rms
is always above the training. 
'''

#Making the N variable to be assigned. Make it 10, 100, 1000
Number = 30
Graph_number = 100

#Making the noise for a gaussian range for ten data points
noise = np.random.normal(0,0.3,Number)

#Making the random x values. Training data
x_values_initial = np.random.uniform(0,1,Number)
x_values = x_values_initial.reshape(-1,1)

#Making the sine function y values. Also training data
y_values_initial = np.sin(2 * np.pi * x_values_initial)
y_data_initial = y_values_initial + noise
y_data = y_data_initial.reshape(-1,1)
#Making graph x values. Testing data. Will be used to calcuate the error
x_graph_values_initial = np.linspace(0,1, Graph_number)
x_graph_values = x_graph_values_initial.reshape(-1,1)

y_graph_values = np.sin(2*np.pi* x_graph_values_initial)
testing_noise = np.random.normal(0,0.3, Graph_number)
y_graph_data_initial = y_graph_values + testing_noise
y_graph_data = y_graph_data_initial.reshape(-1,1)

#Regularization strength needs to be varied. 
lam_bda = np.array([2E-14,2E-13,2E-12,2E-11,2E-10, 2E-9])
#lam_bda = np.array([1E10, 1E15, 1E20, 1E25])
log_lambda = np.log(lam_bda)
#print('log_lambda',log_lambda)


poly9 = PolynomialFeatures(degree=9)
X_poly9 = poly9.fit_transform(x_values)
x_poly9_graph = poly9.fit_transform(x_graph_values)
poly9_reg = LinearRegression()

#Polynomial with Regularization. Doing it in a for loop for multiple lambda values.
#Getting the data of the E_rms in a list
E_rms_testing_data = []
E_rms_training_data = []

def E_rms(train_ridge_y_values, test_ridge_y_values, parameters, train_y_values, test_y_values, lam, train_number, test_number):
    penalty = np.sum(np.square(parameters))
    Error_test = 0.5 * np.sum( np.square(test_ridge_y_values - test_y_values)) + 0.5 * lam * penalty
    E_rms_test = np.sqrt(2 * Error_test / test_number)
    Error_train = 0.5 * np.sum( np.square(train_ridge_y_values - train_y_values)) + 0.5 * lam * penalty
    E_rms_train = np.sqrt(2 * Error_train / train_number)

    return E_rms_train, E_rms_test

#Start making all the fits for all the data points
lam1 = lam_bda[0]
ridge1 = Ridge(alpha = lam1)
ridge1.fit(X_poly9, y_data)
parameters1 = ridge1.coef_
y_values_test_ridge1 = ridge1.predict(x_poly9_graph)
y_values_training_ridge1 = ridge1.predict(X_poly9)
E_rms_train1, E_rms_test1 = E_rms(y_values_training_ridge1, y_values_test_ridge1, parameters1, y_data_initial, y_graph_data_initial, lam1, Number, Graph_number)
E_rms_testing_data.append(E_rms_test1)
E_rms_training_data.append(E_rms_train1)

lam2 = lam_bda[1]
ridge2 = Ridge(alpha = lam2)
ridge2.fit(X_poly9, y_data)
parameters2 = ridge2.coef_
y_values_test_ridge2 = ridge2.predict(x_poly9_graph)
y_values_training_ridge2 = ridge2.predict(X_poly9)
E_rms_train2, E_rms_test2 = E_rms(y_values_training_ridge2, y_values_test_ridge2, parameters2, y_data_initial, y_graph_data_initial, lam2, Number, Graph_number)
E_rms_testing_data.append(E_rms_test2)
E_rms_training_data.append(E_rms_train2)

lam3 = lam_bda[2]
ridge3 = Ridge(alpha = lam3)
ridge3.fit(X_poly9, y_data)
parameters3 = ridge3.coef_
y_values_test_ridge3 = ridge3.predict(x_poly9_graph)
y_values_training_ridge3 = ridge3.predict(X_poly9)
E_rms_train3, E_rms_test3 = E_rms(y_values_training_ridge3, y_values_test_ridge3, parameters3, y_data_initial, y_graph_data_initial, lam3, Number, Graph_number)
E_rms_testing_data.append(E_rms_test3)
E_rms_training_data.append(E_rms_train3)

lam4 = lam_bda[3]
ridge4 = Ridge(alpha = lam4)
ridge4.fit(X_poly9, y_data)
parameters4 = ridge4.coef_
y_values_test_ridge4 = ridge4.predict(x_poly9_graph)
y_values_training_ridge4 = ridge4.predict(X_poly9)
E_rms_train4, E_rms_test4 = E_rms(y_values_training_ridge4, y_values_test_ridge4, parameters4, y_data_initial, y_graph_data_initial, lam4, Number, Graph_number)
E_rms_testing_data.append(E_rms_test4)
E_rms_training_data.append(E_rms_train4)

lam5 = lam_bda[4]
ridge5 = Ridge(alpha = lam5)
ridge5.fit(X_poly9, y_data)
parameters5 = ridge5.coef_
y_values_test_ridge5 = ridge4.predict(x_poly9_graph)
y_values_training_ridge5 = ridge4.predict(X_poly9)
E_rms_train5, E_rms_test5 = E_rms(y_values_training_ridge5, y_values_test_ridge5, parameters5, y_data_initial, y_graph_data_initial, lam5, Number, Graph_number)
E_rms_testing_data.append(E_rms_test5)
E_rms_training_data.append(E_rms_train5)

lam6 = lam_bda[5]
ridge6 = Ridge(alpha = lam6)
ridge6.fit(X_poly9, y_data)
parameters6 = ridge6.coef_
y_values_test_ridge6 = ridge6.predict(x_poly9_graph)
y_values_training_ridge6 = ridge6.predict(X_poly9)
E_rms_train6, E_rms_test6 = E_rms(y_values_training_ridge6, y_values_test_ridge6, parameters6, y_data_initial, y_graph_data_initial, lam6, Number, Graph_number)
E_rms_testing_data.append(E_rms_test6)
E_rms_training_data.append(E_rms_train6)

E_rms_graph_testing = np.array(E_rms_testing_data)
E_rms_graph_training = np.array(E_rms_training_data)


fig, axs = plt.subplots()
plt.title(f'Activity 2b. E_rms vs ln(lambda)')
plt.xlabel('ln(lambda)')
plt.ylabel('E rms')
plt.scatter(log_lambda, E_rms_graph_testing, c='r')
plt.scatter(log_lambda, E_rms_graph_training, c='b')
plt.grid(True)
plt.legend(['testing', 'training'])
plt.show()
plt.close()
