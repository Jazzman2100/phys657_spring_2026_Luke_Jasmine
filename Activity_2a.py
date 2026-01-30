import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures


'''
So, what this python file does is a couple things. First of all, it is set up to make two graphs. 
One graph is the Regularization curve fit on the Ninth Order Polynomial. This is done using the Ridge method in the sklearn library. 
The other graph is the E_rms vs order M graph 
(so technically, despite this being called Activity 2b, it was used to help me answer Activity 1's Analysis questions.)
It also was used to print out what the E_rms values were of the Ninth order polynomial curve fit for 1.b Analysis Question.
'''

#Making the N variable to be assigned. Make it 10, 100, 1000
Number = 100
Graph_number = 100
#Making the noise for a gaussian range for ten data points
noise = np.random.normal(0,0.3,Number)

#Making the random x values. Training data
x_values = np.random.uniform(0,1,Number)

#Making the sine function y values
y_values = np.sin(2 * np.pi * x_values)

y_data = y_values + noise

#Making graph x values
'''
This was the original graph values I used. Just a new validation set.
x_graph_values = np.linspace(0,1,Graph_number).reshape(-1,1)
y_graph_values = np.sin(2*np.pi* x_graph_values).reshape(-1,1)
testing_noise = np.random.normal(0,0.3, Graph_number).reshape(-1,1)
y_graph_data = y_graph_values + testing_noise
'''
#The 25% training and 75% test Method
x_train_values = x_values[0:25].reshape(-1,1)
y_train_values = y_data[0:25].reshape(-1,1)

x_test_values = x_values[25:100].reshape(-1,1)
y_test_values = y_data[25:100].reshape(-1,1)


#Regularization strength
lam_bda = 1

#Making linear regression first.
linear_regression = LinearRegression()
linear_regression.fit(x_train_values, y_train_values)
predict_linear_y_values = linear_regression.predict(x_test_values)
predict_linear_train_y_values = linear_regression.predict(x_train_values)
ridge1 = Ridge(alpha = lam_bda)
ridge1.fit(x_train_values,y_train_values)
predict_linear_ridge_y_values = ridge1.predict(x_test_values)

#Making the other polynomial regressions (2-9)
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(x_train_values)
x_poly2_graph = poly2.fit_transform(x_test_values)
poly2_reg = LinearRegression()
poly2_reg.fit(X_poly2, y_train_values)
predict_poly2_y_values = poly2_reg.predict(x_poly2_graph)
predict_poly2_train_y_values = poly2_reg.predict(X_poly2)
ridge2 = Ridge(alpha = lam_bda)
ridge2.fit(X_poly2, y_train_values)
predict_poly2_ridge_y_values = ridge2.predict(x_poly2_graph)

poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(x_train_values)
x_poly3_graph = poly3.fit_transform(x_test_values)
poly3_reg = LinearRegression()
poly3_reg.fit(X_poly3, y_train_values)
predict_poly3_y_values = poly3_reg.predict(x_poly3_graph)
predict_poly3_train_y_values = poly3_reg.predict(X_poly3)
ridge3 = Ridge(alpha = lam_bda)
ridge3.fit(X_poly3, y_train_values)
predict_poly3_ridge_y_values = ridge3.predict(x_poly3_graph)

poly4 = PolynomialFeatures(degree=4)
X_poly4 = poly4.fit_transform(x_train_values)
x_poly4_graph = poly4.fit_transform(x_test_values)
poly4_reg = LinearRegression()
poly4_reg.fit(X_poly4, y_train_values)
predict_poly4_y_values = poly4_reg.predict(x_poly4_graph)
predict_poly4_train_y_values = poly4_reg.predict(X_poly4)
ridge4 = Ridge(alpha = lam_bda)
ridge4.fit(X_poly4, y_train_values)
predict_poly4_ridge_y_values = ridge4.predict(x_poly4_graph)

poly5 = PolynomialFeatures(degree=5)
X_poly5 = poly5.fit_transform(x_train_values)
x_poly5_graph = poly5.fit_transform(x_test_values)
poly5_reg = LinearRegression()
poly5_reg.fit(X_poly5, y_train_values)
predict_poly5_y_values = poly5_reg.predict(x_poly5_graph)
predict_poly5_train_y_values = poly5_reg.predict(X_poly5)
ridge5 = Ridge(alpha = lam_bda)
ridge5.fit(X_poly5, y_train_values)
predict_poly5_ridge_y_values = ridge5.predict(x_poly5_graph)

poly6 = PolynomialFeatures(degree=6)
X_poly6 = poly6.fit_transform(x_train_values)
x_poly6_graph = poly6.fit_transform(x_test_values)
poly6_reg = LinearRegression()
poly6_reg.fit(X_poly6, y_train_values)
predict_poly6_y_values = poly6_reg.predict(x_poly6_graph)
predict_poly6_train_y_values = poly6_reg.predict(X_poly6)
ridge6 = Ridge(alpha = lam_bda)
ridge6.fit(X_poly6, y_train_values)
predict_poly6_ridge_y_values = ridge6.predict(x_poly6_graph)

poly7 = PolynomialFeatures(degree=7)
X_poly7 = poly7.fit_transform(x_train_values)
x_poly7_graph = poly7.fit_transform(x_test_values)
poly7_reg = LinearRegression()
poly7_reg.fit(X_poly7, y_train_values)
predict_poly7_y_values = poly7_reg.predict(x_poly7_graph)
predict_poly7_train_y_values = poly7_reg.predict(X_poly7)
ridge7 = Ridge(alpha = lam_bda)
ridge7.fit(X_poly7, y_train_values)
predict_poly7_ridge_y_values = ridge7.predict(x_poly7_graph)

poly8 = PolynomialFeatures(degree=8)
X_poly8 = poly8.fit_transform(x_train_values)
x_poly8_graph = poly8.fit_transform(x_test_values)
poly8_reg = LinearRegression()
poly8_reg.fit(X_poly8, y_train_values)
predict_poly8_y_values = poly8_reg.predict(x_poly8_graph)
predict_poly8_train_y_values = poly8_reg.predict(X_poly8)
ridge8 = Ridge(alpha = lam_bda)
ridge8.fit(X_poly8, y_train_values)
predict_poly8_ridge_y_values = ridge8.predict(x_poly8_graph)

poly9 = PolynomialFeatures(degree=9)
X_poly9 = poly9.fit_transform(x_train_values)
x_poly9_graph = poly9.fit_transform(x_test_values)
poly9_reg = LinearRegression()
poly9_reg.fit(X_poly9, y_train_values)
predict_poly9_y_values = poly9_reg.predict(x_poly9_graph)
predict_poly9_train_y_values = poly9_reg.predict(X_poly9)
ridge9 = Ridge(alpha = lam_bda)
ridge9.fit(X_poly9, y_train_values)
predict_poly9_ridge_y_values = ridge9.predict(x_poly9_graph)

E_rms_data = []
E_rms_train = []

#Calculating E_RMS for the other polynomials
Error_testing1 = 0.5 * np.sum( np.square(predict_linear_y_values - y_test_values))
E_rms_testing1 = np.sqrt(2 * Error_testing1 / (Graph_number * .75))
E_rms_data.append(E_rms_testing1)
Error_training1 = 0.5 * np.sum(np.square(predict_linear_train_y_values - y_train_values))
E_rms_training1 = np.sqrt(2 * Error_training1 / (Number * .25))
E_rms_train.append(E_rms_training1)


Error_testing2 = 0.5 * np.sum( np.square(predict_poly2_y_values - y_test_values))
E_rms_testing2 = np.sqrt(2 * Error_testing2 / (Graph_number * .75))
E_rms_data.append(E_rms_testing2)
Error_training2 = 0.5 * np.sum(np.square(predict_poly2_train_y_values - y_train_values))
E_rms_training2 = np.sqrt(2 * Error_training2 / (Number * .25))
E_rms_train.append(E_rms_training2)

Error_testing3 = 0.5 * np.sum( np.square(predict_poly3_y_values - y_test_values))
E_rms_testing3 = np.sqrt(2 * Error_testing3 / (Graph_number * .75))
E_rms_data.append(E_rms_testing3)
Error_training3 = 0.5 * np.sum(np.square(predict_poly3_train_y_values - y_train_values))
E_rms_training3 = np.sqrt(2 * Error_training3 / (Number * .25))
E_rms_train.append(E_rms_training3)

Error_testing4 = 0.5 * np.sum( np.square(predict_poly4_y_values - y_test_values))
E_rms_testing4 = np.sqrt(2 * Error_testing4 / (Graph_number * .75))
E_rms_data.append(E_rms_testing4)
Error_training4 = 0.5 * np.sum(np.square(predict_poly4_train_y_values - y_train_values))
E_rms_training4 = np.sqrt(2 * Error_training4 / (Number * .25))
E_rms_train.append(E_rms_training4)

Error_testing5 = 0.5 * np.sum( np.square(predict_poly5_y_values - y_test_values))
E_rms_testing5 = np.sqrt(2 * Error_testing5 / (Graph_number * .75))
E_rms_data.append(E_rms_testing5)
Error_training5 = 0.5 * np.sum(np.square(predict_poly5_train_y_values - y_train_values))
E_rms_training5 = np.sqrt(2 * Error_training5 / (Number * .25))
E_rms_train.append(E_rms_training5)

Error_testing6 = 0.5 * np.sum( np.square(predict_poly6_y_values - y_test_values))
E_rms_testing6 = np.sqrt(2 * Error_testing6 / (Graph_number * .75))
E_rms_data.append(E_rms_testing6)
Error_training6 = 0.5 * np.sum(np.square(predict_poly6_train_y_values - y_train_values))
E_rms_training6 = np.sqrt(2 * Error_training6 / (Number * .25))
E_rms_train.append(E_rms_training6)

Error_testing7 = 0.5 * np.sum( np.square(predict_poly7_y_values - y_test_values))
E_rms_testing7 = np.sqrt(2 * Error_testing7 / (Graph_number * .75))
E_rms_data.append(E_rms_testing7)
Error_training7 = 0.5 * np.sum(np.square(predict_poly7_train_y_values - y_train_values))
E_rms_training7 = np.sqrt(2 * Error_training7 / (Number * .25))
E_rms_train.append(E_rms_training7)

Error_testing8 = 0.5 * np.sum( np.square(predict_poly8_y_values - y_test_values))
E_rms_testing8 = np.sqrt(2 * Error_testing8 / (Graph_number * .75))
E_rms_data.append(E_rms_testing8)
Error_training8 = 0.5 * np.sum(np.square(predict_poly8_train_y_values - y_train_values))
E_rms_training8 = np.sqrt(2 * Error_training8 / (Number * .25))
E_rms_train.append(E_rms_training8)

Error_testing9 = 0.5 * np.sum( np.square(predict_poly9_y_values - y_test_values))
E_rms_testing9 = np.sqrt(2 * Error_testing9 / (Graph_number * .75))
E_rms_data.append(E_rms_testing9)
Error_training9 = 0.5 * np.sum(np.square(predict_poly9_train_y_values - y_train_values))
E_rms_training9 = np.sqrt(2 * Error_training9 / (Number * .25))
E_rms_train.append(E_rms_training9)

print('9th E_rms', E_rms_testing9)
E_rms = np.array(E_rms_data)
Training_E_rms = np.array(E_rms_train)
order = np.arange(9) + 1

fig, axs = plt.subplots()
plt.title(f'Activity 2. Polynomial fit + Regularization lambda = {lam_bda}')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.scatter(x_values, y_values + noise, c='k')
plt.scatter(x_test_values, predict_poly9_ridge_y_values, c='olive')
plt.grid(True)
plt.show()
plt.close()

fig1, axs1 = plt.subplots()
plt.title('E rms vs Order of Polynomials')
plt.xlabel('Order M')
plt.ylabel('E_rms')
plt.scatter(order, E_rms, c='r')#Test
plt.scatter(order, Training_E_rms, c='b')#Training
plt.legend(['Testing','Training'])
plt.show()
plt.close()

