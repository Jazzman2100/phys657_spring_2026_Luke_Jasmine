import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#Making the N variable to be assigned. Make it 10, 100, 1000
Number = 10

#Making the noise for a gaussian range for ten data points
noise = np.random.normal(0,0.3,Number).reshape(-1,1)

#Making the random x values. Training data
x_values = np.random.uniform(0,1,Number).reshape(-1,1)

#Making the sine function y values
y_values = np.sin(2 * np.pi * x_values).reshape(-1,1)

y_data = y_values + noise

#Making graph x values
x_graph_values = np.linspace(0,1,100).reshape(-1,1)

#Regularization strength
lam_bda = 0.01

#Making linear regression first.
linear_regression = LinearRegression()
#linear_regression.fit(x_values, y_data)
#predict_linear_y_values = linear_regression.predict(x_graph_values)
ridge1 = Ridge(alpha = lam_bda)
ridge1.fit(x_values,y_data)
predict_linear_ridge_y_values = ridge1.predict(x_graph_values)

#Making the other polynomial regressions (2-9)
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(x_values)
x_poly2_graph = poly2.fit_transform(x_graph_values)
poly2_reg = LinearRegression()
#poly2_reg.fit(X_poly2, y_data)
#predict_poly2_y_values = poly2_reg.predict(x_poly2_graph)
ridge2 = Ridge(alpha = lam_bda)
ridge2.fit(X_poly2, y_data)
predict_poly2_ridge_y_values = ridge2.predict(x_poly2_graph)


poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(x_values)
x_poly3_graph = poly3.fit_transform(x_graph_values)
poly3_reg = LinearRegression()
#poly3_reg.fit(X_poly3, y_data)
#predict_poly3_y_values = poly3_reg.predict(x_poly3_graph)
ridge3 = Ridge(alpha = lam_bda)
ridge3.fit(X_poly3, y_data)
predict_poly3_ridge_y_values = ridge3.predict(x_poly3_graph)

poly4 = PolynomialFeatures(degree=4)
X_poly4 = poly4.fit_transform(x_values)
x_poly4_graph = poly4.fit_transform(x_graph_values)
poly4_reg = LinearRegression()
#poly4_reg.fit(X_poly4, y_data)
#predict_poly4_y_values = poly4_reg.predict(x_poly4_graph)
ridge4 = Ridge(alpha = lam_bda)
ridge4.fit(X_poly4, y_data)
predict_poly4_ridge_y_values = ridge4.predict(x_poly4_graph)

poly5 = PolynomialFeatures(degree=5)
X_poly5 = poly5.fit_transform(x_values)
x_poly5_graph = poly5.fit_transform(x_graph_values)
poly5_reg = LinearRegression()
#poly5_reg.fit(X_poly5, y_data)
#predict_poly5_y_values = poly5_reg.predict(x_poly5_graph)
ridge5 = Ridge(alpha = lam_bda)
ridge5.fit(X_poly5, y_data)
predict_poly5_ridge_y_values = ridge5.predict(x_poly5_graph)

poly6 = PolynomialFeatures(degree=6)
X_poly6 = poly6.fit_transform(x_values)
x_poly6_graph = poly6.fit_transform(x_graph_values)
poly6_reg = LinearRegression()
#poly6_reg.fit(X_poly6, y_data)
#predict_poly6_y_values = poly6_reg.predict(x_poly6_graph)
ridge6 = Ridge(alpha = lam_bda)
ridge6.fit(X_poly6, y_data)
predict_poly6_ridge_y_values = ridge6.predict(x_poly6_graph)

poly7 = PolynomialFeatures(degree=7)
X_poly7 = poly7.fit_transform(x_values)
x_poly7_graph = poly7.fit_transform(x_graph_values)
poly7_reg = LinearRegression()
#poly7_reg.fit(X_poly7, y_data)
#predict_poly7_y_values = poly7_reg.predict(x_poly7_graph)
ridge7 = Ridge(alpha = lam_bda)
ridge7.fit(X_poly7, y_data)
predict_poly7_ridge_y_values = ridge7.predict(x_poly7_graph)

poly8 = PolynomialFeatures(degree=8)
X_poly8 = poly8.fit_transform(x_values)
x_poly8_graph = poly8.fit_transform(x_graph_values)
poly8_reg = LinearRegression()
#poly8_reg.fit(X_poly8, y_data)
#predict_poly8_y_values = poly8_reg.predict(x_poly8_graph)
ridge8 = Ridge(alpha = lam_bda)
ridge8.fit(X_poly8, y_data)
predict_poly8_ridge_y_values = ridge8.predict(x_poly8_graph)

poly9 = PolynomialFeatures(degree=9)
X_poly9 = poly9.fit_transform(x_values)
x_poly9_graph = poly9.fit_transform(x_graph_values)
poly9_reg = LinearRegression()
#poly9_reg.fit(X_poly9, y_data)
#predict_poly9_y_values = poly9_reg.predict(x_poly9_graph)
ridge9 = Ridge(alpha = lam_bda)
ridge9.fit(X_poly9, y_data)
predict_poly9_ridge_y_values = ridge9.predict(x_poly9_graph)

fig, axs = plt.subplots()
plt.title(f'Activity 2. Polynomial fit + Regularization lambda = {lam_bda}')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.scatter(x_values, y_values + noise, c='k')
plt.plot(x_graph_values, predict_linear_ridge_y_values, c='b')
plt.plot(x_graph_values, predict_poly2_ridge_y_values, c='r')
plt.plot(x_graph_values, predict_poly3_ridge_y_values, c='g')
plt.plot(x_graph_values, predict_poly4_ridge_y_values, c='c')
plt.plot(x_graph_values, predict_poly5_ridge_y_values, c='navy')
plt.plot(x_graph_values, predict_poly6_ridge_y_values, c='gold')
plt.plot(x_graph_values, predict_poly7_ridge_y_values, c='m')
plt.plot(x_graph_values, predict_poly8_ridge_y_values, c='orange')
plt.plot(x_graph_values, predict_poly9_ridge_y_values, c='olive')
plt.legend(['data', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th'])


plt.grid(True)
plt.show()
