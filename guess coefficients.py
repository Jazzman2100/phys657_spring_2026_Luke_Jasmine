import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from scipy.optimize import curve_fit
import math

size = 100
sigma = 0.3

# 1. Define the model function (a-i are parameters p[0]-p[8])
def poly_func(x, a, b, c, d, e, f, g, h, i):
    return a*x + b*x**2 + c*x**3 + d*x**4 + e*x**5 + f*x**6 + g*x**7 + h*x**8 + i*x**9

# 2. Sample Data (replace with your actual data)
x_data = np.linspace(0, 1, size)

y1 = np.sin(2*math.pi*x_data)
y2 = np.random.normal(loc=0, scale=sigma, size=size)
y_data = y1 + y2

ln_lambda = np.linspace(-40, -20, 21)
lambdas = np.exp(ln_lambda)
e_rms_list = []

for lam in lambdas:
    def objective(w):
        y_fit = poly_func(x_data, *w)
        error_sum = 1/2 * np.sum((y_fit-y_data)**2)
        penalty = 1/2 * lam * np.sum(w**2)
        return error_sum + penalty
    res = minimize(objective, x0 = np.zeros(9))

    e_w = res.fun
    e_rms = np.sqrt(2*e_w/size)
    e_rms_list.append(e_rms)


# 3. Initial Guess: Start with zeros or small numbers
initial_guess = [0.0] * 9  # [a, b, c, d, e, f, g, h, i]

# 4. Fit the curve
popt, pcov = curve_fit(poly_func, x_data, y_data, p0=initial_guess)

print("Estimated parameters (a-i):", popt)

x_smooth = np.linspace(0, 1, 200)
# Use *popt to "unpack" the optimized parameters into the function
y_fit = poly_func(x_smooth, *popt)

# 5. Plotting
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color='blue', label='Actual Data')
plt.plot(x_smooth, y_fit, color='red', linewidth=2, label='9th Degree Fit') # Added fit
plt.title('9th Degree Polynomial Fit')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(ln_lambda, e_rms_list, 'o-', color='red', label='$E_{RMS}$')
plt.title('Regularization Analysis: $\ln(\lambda)$ vs $E_{RMS}$')
plt.xlabel('$\ln(\lambda)$')
plt.ylabel('$E_{RMS}$')
plt.grid(True)
plt.legend()
plt.show()